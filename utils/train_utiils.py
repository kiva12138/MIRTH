"""Utils for training/fine-tuning scripts."""

import torch

from config.config_vla import ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, ACTION_TOKEN_END_IDX, ACTION_REASON_TOKEN_BEGIN_IDX, ACTION_REASON_TOKEN_END_IDX


def get_action_tokens_mask(token_ids):
    valid_token_mask = token_ids != IGNORE_INDEX
    action_token_mask_1 = token_ids >= ACTION_TOKEN_BEGIN_IDX
    action_token_mask_2 = token_ids <= ACTION_TOKEN_END_IDX
    action_token_mask = action_token_mask_1 & action_token_mask_2
    valid_action_token_mask = valid_token_mask & action_token_mask
    return valid_action_token_mask


def get_reasoning_tokens_mask(token_ids):
    valid_token_mask = token_ids != IGNORE_INDEX
    reason_token_mask_1 = token_ids >= ACTION_REASON_TOKEN_BEGIN_IDX
    reason_token_mask_2 = token_ids <= ACTION_REASON_TOKEN_END_IDX
    reason_token_mask = reason_token_mask_1 & reason_token_mask_2
    valid_reason_token_mask = valid_token_mask & reason_token_mask
    return valid_reason_token_mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy()))
    true_continuous_actions = torch.tensor(action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy()))
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss
