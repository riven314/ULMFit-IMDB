from fastai.text import *

bs = 48
path = untar_data(URLs.IMDB)

#data_lm = (TextList.from_folder(path).filter_by_folder(include = ['train', 'test', 'unsup']).split_by_rand_pct(0.1).label_for_lm().databunch(bs = bs))
data_lm = load_data(path, 'data_lm.pkl')
print('IMBD databunch loaded!')

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult = 0.3)
learn.load('fit_head')
learn.unfreeze()
learn.fit_one_cycle(8, 1e-3, moms = (0.8, 0.7))
learn.save('fine_tuned')
print('Fine-tuned LM model saved!')
