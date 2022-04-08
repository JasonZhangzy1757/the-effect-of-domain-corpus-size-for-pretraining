import random as rng
from collections import namedtuple
from tokenizers import BertWordPieceTokenizer 

MaskedLmInstance = namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_ids(token_ids: list, 
                         source_ids: list, 
                         tokenizer: BertWordPieceTokenizer, 
                         masked_lm_prob: float=0.15, 
                         max_predictions_per_seq: int=80):

  """
  Creates the predictions for the masked LM objective.

  Args:
  token_ids: list of tokenized ids (likely a sequence of len 512).
  source_ids: source ids from tokenizer.  Len should equal len of tokenizer vocabulary.
  tokenizer: currently only support Class BertWordPieceTokenizer

  """

  cand_indexes = []
  for (i, token) in enumerate(token_ids):
    if token == 2 or token == 3:
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if len(cand_indexes) >= 1 and tokenizer.id_to_token(token).startswith('##'):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(token_ids)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(token_ids) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = tokenizer.token_to_id('[MASK]')
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = token_ids[index]
        # 10% of the time, replace with random word
        else:
          masked_token = source_ids[rng.randint(0, len(source_ids) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=token_ids[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return output_tokens