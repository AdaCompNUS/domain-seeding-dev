import torch


def to_batch(images, label, device):
    images = torch.FloatTensor(images).unsqueeze(0).to(device)
    label = torch.FloatTensor(label).unsqueeze(0).to(device)
    return images, label


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    # torch.autograd.set_detect_anomaly(True)
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())
