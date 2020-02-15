# ULMFit-IMDB
extended analysis on ULMFit modeling from lesson 4, Practical Deep Learning for Coders (fast.ai)

# OBSERVATION
With fine-tuning on language model, I received 93%+ validation accuracy on IMDB dataset. It contrasts with 90%+ I received without fine-tuning on language model. When training without fine-tuning on language model, the validation accuracy starts at ~50%, but the accuracy progressively increase as I fine-tune with more layer unfreezed. In fact, the net effect of training with more unfreezed layers is similar to fine-tuning on the language model. 
