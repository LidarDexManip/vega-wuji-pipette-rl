"""(11-09) Action smoothness on the policy MEAN, as a PPO loss term (CAPS temporal term, Mysore et al. 2021), not as a reward.

Why not a reward: a rate penalty in the reward sees the SAMPLED action, i.e. also the exploration noise. With PPO's Gaussian noise
(std 0.1 normalised action at the start) the noise alone costs far more than the policy's own jitter (zero residual + noise 0.1:
action_rate 24.7, torque_rate_arm 21.5, torque_rate_finger 12.4 raw units; rr_T1k_tcq_s5 weights -> -68 per step against ~ +21 of
tracking), so the trainer collapses its std (0.10 -> 0.03) and prefers short episodes. Here the term acts on mu(s), the actor's
deterministic output, through the gradient: exploration noise is not penalised and the reward stays as it was.

  L = mean over sampled (t, t+1) pairs of the same episode of min( mean_actuators ((mu(s_t+1) - mu(s_t)) / norm)^2 , cap )
  norm 0.01 normalised action (the unit of action_rate_norm: 1 = a change of 0.01 on every actuator); cap 25 (larger changes, e.g. at a
  contact event, cost a constant and get no gradient).

Plugged into rsl_rl's PPO (5.5.x) through its symmetry hook: PPO.update calls alg.symmetry.augment_batch(...) and, when
alg.symmetry.use_mirror_loss, adds alg.symmetry.mirror_loss_coeff x alg.symmetry.compute_loss(actor, batch, n) to every mini-batch
loss (logged as loss 'symmetry'). rsl_rl stays unmodified. Pairs come from the whole rollout storage of the iteration (the shuffled
mini-batch has no consecutive steps); a pair whose step t ended the episode (dones[t]) is skipped.

Config (train.py): "caps": {"coeff": C, "norm": 0.01, "cap": 25, "pairs": 8192}"""
import torch


class ActionSmoothness:
    use_data_augmentation = False
    use_mirror_loss = True

    def __init__(self, alg, coeff, norm=.01, cap=25., pairs=8192, seed=0):
        self.alg = alg; self.mirror_loss_coeff = float(coeff); self.norm = float(norm); self.cap = float(cap); self.pairs = int(pairs)
        self.g = torch.Generator(device=alg.device); self.g.manual_seed(seed)

    def augment_batch(self, batch, original_batch_size):
        return None

    def sample_pairs(self):
        st = self.alg.storage; T, N = st.dones.shape[:2]
        t = torch.randint(0, T-1, (self.pairs,), device=st.dones.device, generator=self.g)
        e = torch.randint(0, N, (self.pairs,), device=st.dones.device, generator=self.g)
        ok = st.dones[t, e].reshape(-1) == 0
        return st.observations[t, e], st.observations[t+1, e], ok

    def loss_on(self, actor, o0, o1, ok):
        K = ok.shape[0]; mu = actor(torch.cat([o0, o1], 0))
        d = (((mu[K:]-mu[:K]).float()/self.norm)**2).mean(1).clamp(max=self.cap)
        return (d*ok).sum()/ok.sum().clamp(min=1)

    def compute_loss(self, actor, batch, original_batch_size):
        return self.loss_on(actor, *self.sample_pairs())


def attach(alg, cfg):
    """cfg: the config's 'caps' dict (or None / {} / coeff 0: nothing attached). Returns the hook or None."""
    if not cfg or not float(cfg.get('coeff', 0)): return None
    if alg.symmetry is not None: raise RuntimeError('caps uses the PPO symmetry hook, which is already in use')
    if alg.actor.is_recurrent: raise RuntimeError('caps: recurrent actors not supported')
    alg.symmetry = ActionSmoothness(alg, cfg['coeff'], cfg.get('norm', .01), cfg.get('cap', 25.), cfg.get('pairs', 8192), cfg.get('seed', 0))
    return alg.symmetry
