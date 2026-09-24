"""Local episode fine-tuning 
 
Takes an encoder alteady initialised and fune-tunes it in place on  
one episode's (D_S, D_syn) pair for 8 epochs, producing thete_local. 
A fresh episode classifier F_c is created here and discarded on return. 
 
L_total = L_con^s + L_con^u + L_s + L_adv + lambda * L_margin 
""" 
 
from collections.abc import Iterator 
import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
 
from ..data.episodes import Episode 
from ..losses import ( 
    confidence_margin_loss, 
    open_set_adversarial_loss, 
    unsupervised_contrastive_loss, 
    supervised_prototype_contrastive_loss 
) 
from ..models import EpisodeClassifier 
 
 
def _cycle(loader: DataLoader) -> Iterator: 
    """Endlessly repeat a loader, reshuffling each pass.""" 
    while True: 
        yield from loader 
 
# This function takes an encoder and an episode, and fine-tunes the encoder on that 
# episode for a specified number of epochs. It returns a dictionary containing the  
# final loss values and the fine-tuned encoder. 
def train_local_episode( 
        encoder,  
        episode: Episode,  
        cfg,  
        device: torch.device, 
        verbose: bool = True, 
) -> dict: 
    """Fine-tune `encoder` in place on one episode. 
 
    Args: 
        encoder: DINOViTBackbone already holding theta_global^{g-1}. 
            MUTATED -- the caller is responsible for having made a copy. 
        episode: from EpisodeSampler. 
        cfg: full config. 
        device: target device. 
        verbose: print a per-epoch line. 
 
    Returns: 
        History dict: each key maps to a list of per-epoch mean values. 
    """ 
    ec = cfg.episodic 
    lc = cfg.losses 
 
    encoder.to(device) 
 
    # Fresh classifier: |Y_s^eg| + 1 outputs. |Y_s^eg| changes per episode 
    # So there is nothing to carry over from one episode to the other. 
    classifier = EpisodeClassifier( 
        in_features=encoder.embed_dim, 
        num_known_classes=episode.num_known_classes, 
    ).to(device) 
 
    # Common data loader arguments 
    common = dict( 
        batch_size = ec.batch_size, 
        shuffle = True,  
        drop_last = True, 
        num_workers = 0, 
        pin_memory = (device.type == "cuda") 
    ) 
    source_loader = DataLoader(episode.source_dataset, **common) 
    synth_loader = DataLoader(episode.synthetic_dataset, **common) 
 
    if len(source_loader) == 0 or len(synth_loader) == 0: 
        raise ValueError("Episode has no data in source or synthetic dataset.") 
 
    # D_syn holds all of Y_s while D_S holds only Y_s^eg, so the loaders differ in length. 
    # One epoch is a full pass over the LONGER one, cucling the shorter, so neither is starved. 
    steps_per_epoch = max(len(source_loader), len(synth_loader)) 
    total_steps = steps_per_epoch * ec.local_epochs 
 
    task_params = list(encoder.trainable_parameters()) 
    task_optimizer = torch.optim.SGD( 
        task_params, lr=ec.lr, momentum=ec.momentum, weight_decay=ec.weight_decay 
    ) 
    task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
        task_optimizer, T_max=ec.local_epochs, eta_min=ec.lr * 1e-3 
    ) 
    optim_c = torch.optim.SGD( 
        classifier.parameters(), lr=ec.lr, 
        momentum=ec.momentum, weight_decay=ec.weight_decay, 
    ) 
    classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
        optim_c, T_max=ec.local_epochs, eta_min=ec.lr * 1e-3 
    ) 
    history = { 
    "total":  [], 
    "con_s": [], 
    "con_u": [], 
    "ce_s": [], 
    "adv": [], 
    "margin": [], 
    "p_unknown": [], 
    "feat_norm": [], 
    } 
     
    encoder.train() 
    classifier.train() 
 
    for epoch in range(ec.local_epochs):  
        source_iter = _cycle(source_loader)  
        synth_iter = _cycle(synth_loader)  
        running = dict.fromkeys(history, 0.0)  
  
        for _ in range(steps_per_epoch):  
            (xs1, xs2), ys = next(source_iter)  
            (xu1, xu2), _ = next(synth_iter)  
  
            xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)  
            xu1, xu2 = xu1.to(device), xu2.to(device)  
  
            # Forward pass  
            fs1, fs2 = encoder(xs1), encoder(xs2)  
            ft1, ft2 = encoder(xu1), encoder(xu2)  
  
            # L_adv : adversarial loss, open-set  
            loss_adv = open_set_adversarial_loss(  
                classifier(ft1.detach()),  
                alpha=lc.adv_alpha 
            )  
  
            optim_c.zero_grad(set_to_none=True)  
            loss_adv.backward()  
            optim_c.step()  
  
            # L_con^s : Used for better clustering of known classes  
            loss_con_s = supervised_prototype_contrastive_loss( 
                fs1,  
                ys,  
                num_classes=episode.num_known_classes,  
                temperature=lc.temperature 
            )  
  
            # L_con^u : Used for better clustering of unknown classes   
            loss_con_u = unsupervised_contrastive_loss( 
                torch.cat([fs1, ft1], dim=0), 
                torch.cat([fs2, ft2], dim=0), 
                temperature=lc.temperature 
            )  
  
            # L_s : source classification loss, cross-entropy  
            loss_ce_s = F.cross_entropy(classifier(fs1), ys)  
  
            # L_margin : confidence margin loss, open-set  
            logits_t = classifier(ft1)  
            loss_margin = confidence_margin_loss( 
                logits_t, 
                margin=lc.margin_m 
            )  
  
            with torch.no_grad():  
                probs_t = torch.softmax(logits_t, dim=1)  
                p_unknown = probs_t[:, -1].mean().item()  
                feat_norm = ft1.norm(dim=1).mean().item()  
  
            # Domain adaptation loss  
            loss_da = loss_ce_s + lc.lambda_da * loss_margin  
            loss_total = ( 
                lc.w_con_u * loss_con_u 
                + lc.w_con_s * loss_con_s 
                + lc.mu_da * loss_da 
            )  
  
            task_optimizer.zero_grad(set_to_none=True)  
            optim_c.zero_grad(set_to_none=True)  
            loss_total.backward()  
  
            if ec.grad_clip:  
                torch.nn.utils.clip_grad_norm_(task_params, ec.grad_clip)  
  
            task_optimizer.step()  
            optim_c.step()  
  
            running["total"] += loss_total.item()  
            running["con_s"] += loss_con_s.item()  
            running["con_u"] += loss_con_u.item()  
            running["ce_s"] += loss_ce_s.item()  
            running["adv"] += loss_adv.item()  
            running["margin"] += loss_margin.item()  
            running["p_unknown"] += p_unknown  
            running["feat_norm"] += feat_norm  
  
        task_scheduler.step()  
        classifier_scheduler.step()  
  
        for key in history:  
            history[key].append(running[key] / steps_per_epoch)  
  
        if verbose:  
            print(f"Epoch {epoch+1}/{ec.local_epochs} | "  
                  f"Total: {history['total'][-1]:.4f} | "  
                  f"L_con^s: {history['con_s'][-1]:.4f} | "  
                  f"L_con^u: {history['con_u'][-1]:.4f} | "  
                  f"L_ce^s: {history['ce_s'][-1]:.4f} | "  
                  f"L_adv: {history['adv'][-1]:.4f} | "  
                  f"L_margin: {history['margin'][-1]:.4f} | "  
                  f"p_unk: {history['p_unknown'][-1]:.4f} | "  
                  f"|f|: {history['feat_norm'][-1]:.4f} | "  
                  f"LR: {task_scheduler.get_last_lr()[0]:.6f}")  
 
    return history