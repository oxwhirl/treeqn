import torch
import torch.nn.functional as F
from treeqn.utils.pytorch_utils import cudify
import os
from datetime import datetime

# discounting reward sequences
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_seq_mask(mask):
    max_i = torch.max(mask, 0)[1]
    if (mask[max_i] == 1).all():
        mask[int(max_i):].fill_(1)
    return (1 - mask).unsqueeze(1)

# some utilities for interpreting the trees we return
def build_sequences(sequences, masks, nenvs, nsteps, depth, return_mask=False, offset=0):
    # sequences are bs x size, containing e.g. rewards, actions, state reps
    # returns bs x depth x size processed sequences with a sliding window set by 'depth', padded with 0's
    # if return_mask=True also returns a mask showing where the sequences were padded
    # This can be used to produce targets for tree outputs, from the true observed sequences
    tmp_masks = torch.from_numpy(masks.reshape(nenvs, nsteps).astype(int))
    tmp_masks = F.pad(tmp_masks, (0, depth+offset, 0, 0), mode="constant", value=1).data

    sequences = [s.view(nenvs, nsteps, -1) for s in sequences]
    if return_mask:
        mask = torch.ones_like(sequences[0]).float()
        sequences.append(mask)
    sequences = [F.pad(s, (0, 0, 0, depth+offset, 0, 0), mode="constant", value=0).data for s in sequences]
    proc_sequences = []
    for seq in sequences:
        proc_seq = []
        for env in range(seq.shape[0]):
            for t in range(nsteps):
                seq_done_mask = make_seq_mask(tmp_masks[env, t+offset:t+offset+depth].clone())
                proc_seq.append(seq[env, t+offset:t+offset+depth, :].float() * seq_done_mask.float())
        proc_sequences.append(torch.stack(proc_seq))
    return proc_sequences


def get_paths(tree, actions, batch_size, num_actions):
    # gets the parts of the tree corresponding to actions taken
    action_indices = cudify(torch.zeros_like(actions[:,0]).long())
    output = []
    for i, x in enumerate(tree):
        action_indices = action_indices * num_actions + actions[:, i]
        batch_indices = cudify(torch.arange(0, batch_size).long() * x.size(0) / batch_size) + action_indices
        output.append(x[batch_indices])
    return output


def get_subtree(tree, actions, batch_size, num_actions):
    # gets the subtree corresponding to actions taken
    action_indices = actions[:,0]
    output = []
    for i, x in enumerate(tree[1:]):
        batch_starts = cudify(torch.arange(0, batch_size) * x.size(0) / batch_size)
        indices = []
        for b in range(batch_size):
            indices.append(cudify(torch.arange(action_indices[b] * num_actions**i, (action_indices[b]+1) * num_actions**i)) + batch_starts[b])
        indices = torch.cat(indices).long()
        output.append(x[indices])
    return output


def time_shift_tree(tree, nenvs, nsteps, offset):
    # shifts the tree by an offset
    output = []
    for i, x in enumerate(tree):
        x = x.view(nenvs, nsteps, -1)
        if offset >= 0:
            x = x[:, offset:, :]
        else:
            x = x[:, :offset, :]
        output.append(x)
    return output



def get_timestamped_dir(path, name=None, link_to_latest=False):
    """Create a directory with the current timestamp."""
    current_time = datetime.now().strftime("%y-%m-%d/%H-%M-%S-%f")
    dir = path + "/" + current_time + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if name is not None:
        if os.path.exists(path + "/" + name):
            os.remove(path + "/" + name)
        os.symlink(current_time, path + "/" + name, target_is_directory=True)
    if link_to_latest:
        if os.path.exists(path + "/latest"):
            os.remove(path + "/latest")
        os.symlink(current_time, path + "/latest", target_is_directory=True)
    return dir


def append_scalar(run, key, val):
    if key in run.info:
        run.info[key].append(val)
    else:
        run.info[key] = [val]


def append_list(run, key, val):
    if key in run.info:
        run.info[key].extend(val)
    else:
        run.info[key] = [val]
