import torch

def update_sim(sim_tensor, interaction_mask, pred_prob, num_classes):
    """
    Update the Sequential Interaction Memory (SIM) tensor for one sample.

    Args:
        sim_tensor (torch.Tensor): Current SIM, shape (2*D*N, H, W)
        interaction_mask (torch.Tensor): User clicks (one-hot), shape (N, H, W)
        pred_prob (torch.Tensor): Predicted probability maps, shape (N, H, W)
        num_classes (int): Number of segmentation classes (including background)

    Returns:
        torch.Tensor: Updated SIM tensor
    """
    device = sim_tensor.device
    interaction_mask = interaction_mask.to(device)
    pred_prob = pred_prob.to(device)

    total_channels = sim_tensor.shape[0]
    D = total_channels // (2 * num_classes)

    # Figure out which round we're at = count filled rounds
    # Each round = num_classes clicks + num_classes probs
    for round_idx in range(D):
        click_ch_start = round_idx * num_classes
        click_ch_end   = click_ch_start + num_classes

        # Check if this round's click channels are empty
        if torch.all(sim_tensor[click_ch_start:click_ch_end] == 0):
            # Insert clicks
            sim_tensor[click_ch_start:click_ch_end] = interaction_mask
            # Insert probs (after the D*N clicks block)
            prob_ch_start = D * num_classes + round_idx * num_classes
            prob_ch_end   = prob_ch_start + num_classes
            sim_tensor[prob_ch_start:prob_ch_end] = pred_prob
            break

    return sim_tensor
