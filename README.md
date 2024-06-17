# Read Me

## News

- This paper has been accepted to ICIC 2024 (Oral) (2024.5.13)
- We have updated the paper uploaded to ArXiv: https://arxiv.org/abs/2404.03921  (2024.5.16)

***

## Quick Start

- Install Dependencies

  ```bash
  pip install -r requirements.txt
  ```

- Download Data

  ```bash
  cd SentEval/data/downstream/
  bash download_dataset.sh
  cd -
  cd ./data
  bash download_nli.sh
  cd -
  ```

- Python version 3.9.18

***

## Acknowledgement

- Our code is based on PromptEOL

## Friendship Link

- [CoT-BERT](https://github.com/ZBWpro/CoT-BERT): State-of-the-Art :star2: <u>unsupervised</u> sentence representation scheme based on <u>discriminative</u> pre-trained language models (BERT, RoBERTa). [CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought](https://arxiv.org/abs/2309.11143)
- [STS-Regression](https://github.com/ZBWpro/STS-Regression): State-of-the-Art :star2: <u>supervised</u> sentence representation scheme based on <u>discriminative</u> pre-trained language models (BERT, RoBERTa). [Advancing Semantic Textual Similarity Modeling: A Regression Framework with Translated ReLU and Smooth K2 Loss](https://arxiv.org/abs/2406.05326)
- [PretCoTandKE](https://github.com/ZBWpro/PretCoTandKE): State-of-the-Art :star2: ​<u>direct inference</u> scheme for sentence embeddings based on <u>generative</u> pre-trained language models (OPT, LLaMA, LLaMA2, Mistral). [Simple Techniques for Enhancing Sentence Embeddings in Generative Language Models](https://arxiv.org/abs/2404.03921)
- [Pcc-tuning](https://github.com/ZBWpro/Pcc-tuning): State-of-the-Art :star2: ​<u>supervised</u> sentence representation scheme based on <u>generative</u> pre-trained language models (OPT, LLaMA, LLaMA2, Mistral). [Pcc-tuning: Breaking the Contrastive Learning Ceiling in Semantic Textual Similarity](https://arxiv.org/abs/2406.09790)