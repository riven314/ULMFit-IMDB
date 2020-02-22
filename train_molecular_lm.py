import pandas as pd

from fastai import *
from fastai.text import *
from molecule_utils import *

bs = 128
path = Path('results/MSPM')
print(f'batch size: {bs}')
print(f'save path: {path}')

train = pd.read_csv('data/MSPM/ChemBL-LM_train.csv')
valid = pd.read_csv('data/MSPM/ChemBL-LM_val.csv')
# for testing
#train_aug = train.head(1000)
#valid_aug = valid.head(500)
train_aug = smiles_augmentation(train, 4)
valid_aug = smiles_augmentation(valid, 4)
print(f'train data: {train_aug.shape[0]}')
print(f'valid data: {valid_aug.shape[0]}')

if os.path.isfile(path/'MSPM_databunch.pkl'):
    data_lm = load_data(path, 'MSPM_databunch.pkl', bs = bs)
    print('archive databunch is found, preloaded')
else:
    tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus = 6, pre_rules = [], post_rules = [])
    data_lm = TextLMDataBunch.from_df(path, train_aug, valid_aug,
                                      bs = bs, tokenizer = tok,
                                      chunksize = 50000, 
                                      text_cols = 0, max_vocab = 60000,
                                      include_bos = False)

    data_lm.save(f'MSPM_databunch.pkl')
    print('databunch loaded')

learner = language_model_learner(data_lm, AWD_LSTM, drop_mult = 1, pretrained = False)
lr = (3e-3 * bs) / 48

# unfreeze all layers and train
print('start training')
learner.unfreeze()
learner.fit_one_cycle(10, lr, moms = (0.8, 0.7))
# save model and vocab
learner.save('MSPM_wt', with_opt = False)
learner.data.vocab.save(path / 'MSPM_vocab.pkl')

