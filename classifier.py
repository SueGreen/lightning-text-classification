# -*- coding: utf-8 -*-
import logging as log
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorboard as tb
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors
from transformers import AutoModel, AutoTokenizer
from torchmetrics import Accuracy

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929


class Classifier(pl.LightningModule):
    """
    Sample model to use a Transformer model to classify sentences / phrases.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
            self.classifier = classifier_instance
            # Label Encoder
            products_df = pd.read_csv(self.classifier.hparams.products_csv)
            products_df.rename(columns={"product_title": "text", "category_id": "label"}, inplace=True)
            self.label_encoder = LabelEncoder(
                products_df.label.astype(str).unique().tolist(),
                reserved_labels=[]
            )
            self.categories_df = pd.read_csv(self.classifier.hparams.categories_csv)

        def read_csv(self, path: str) -> list:
            """ Reads a comma separated value file.

            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
            df = pd.read_csv(path)
            df.rename(columns={"product_title": "text", "category_id": "label"}, inplace=True)
            df = df[["text", "label"]]
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(str)
            return df.to_dict("records")

        def train_dataloader(self) -> DataLoader:
            """ Function that loads the train set. """
            self._train_dataset = self.read_csv(self.classifier.hparams.train_csv)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.classifier.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.classifier.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            """ Function that loads the validation set. """
            self._dev_dataset = self.read_csv(self.classifier.hparams.dev_csv)
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.classifier.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.classifier.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            """ Function that loads the test set. """
            self._test_dataset = self.read_csv(self.classifier.hparams.test_csv)
            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.classifier.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.classifier.hparams.loader_workers,
            )

    def __init__(self, hparams: Namespace) -> None:
        super(Classifier, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = hparams.batch_size

        # Build Data module
        self.data = self.DataModule(self)

        # build model
        self.__build_model()

        # Loss criterion and main metric initialization.
        self.__build_loss()
        self.__build_metrics()

        if hparams.n_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.bert = AutoModel.from_pretrained(
            self.hparams.encoder_model, output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        if self.hparams.encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.encoder_model)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features * 2),
            nn.Tanh(),
            nn.Linear(self.encoder_features * 2, self.encoder_features),
            nn.Tanh(),
            nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),
        )

    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss()

    def __build_metrics(self):
        """ Initializes the loss function/s. """
        self._metric = Accuracy()

    def _unfreeze_encoder_layer(self):
        self.n_unfreezed_layers += 1
        for module_name in [list(self.bert.modules())[-self.n_unfreezed_layers]]:
            if len(list(module_name.parameters())) > 0:
                for param in module_name.parameters():
                    param.requires_grad = True

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            self._unfreeze_encoder_layer()
            self._frozen = False
        else:
            # if number of epochs for one layer fine-tuning is reached
            # and layer is allowed to be unfreezed, then unfreeze the next layer
            if (self.current_epoch - self.hparams.n_frozen_epochs) % \
                    self.hparams.n_epochs_per_layers_unfreezing == 0 and \
                    self.n_unfreezed_layers < self.hparams.max_n_unfreezed_layers:
                self._unfreeze_encoder_layer()

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True
        self.n_unfreezed_layers = 0

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            index = self.data.categories_df.category_id == int(predicted_labels[0])
            predicted_label = self.data.categories_df[index].iloc[0]
            sample["predicted_label"] = predicted_label["category_title"]
            sample["predicted_label_path"] = predicted_label["category_path"]

        return sample

    def forward(self, input_ids, token_type_ids, attention_mask):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = input_ids
        mask = attention_mask

        # Run BERT model.
        word_embeddings = self.bert(tokens, token_type_ids, mask)[0]
        sentence_embeddings = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentence_embeddings = sentence_embeddings / sum_mask

        return {"sentence_embeddings": sentence_embeddings, "logits": self.classification_head(sentence_embeddings)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def metric(self, model_out: dict, targets: dict) -> torch.tensor:
        predictions = torch.argmax(model_out["logits"], dim=1)
        return self._metric(predictions, targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        inputs = self.tokenizer(sample["text"], padding=True, truncation=True, return_tensors="pt", verbose=False)

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def step(self, batch: tuple, training_phase: str, prog_bar=False):
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss = self.loss(model_out, targets)
        self.log(f"a/loss/{training_phase}", loss, on_step=True, on_epoch=True, prog_bar=prog_bar, logger=True)
        metric = self.metric(model_out, targets)
        self.log(f"a/acc/{training_phase}", metric, on_step=True, on_epoch=True, prog_bar=prog_bar, logger=True)
        result = dict({"loss": loss, "metric": metric})
        return result

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """ 
        Runs on training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        return self.step(batch, training_phase="train")

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """
        Runs on validation step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of the validation dataloader.
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        return self.step(batch, training_phase="val")

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        """
            Runs on testing step. This usually consists in the forward function followed
                by the loss function.

            :param batch: The output of the validation dataloader.
            :param batch_nb: Integer displaying which batch this is

            Returns:
                - dictionary containing the loss and the metrics to be added to the lightning logger.
            """
        return self.step(batch, training_phase="test")

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": float(self.hparams.encoder_learning_rate),
            },
        ]
        optimizer = optim.Adam(parameters, lr=float(self.hparams.learning_rate))
        return [optimizer], []

    def on_train_epoch_end(self, unused=None):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.hparams.n_frozen_epochs:
            self.unfreeze_encoder()
