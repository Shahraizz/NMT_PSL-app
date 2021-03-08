from configparser import ConfigParser
from bpemb import BPEmb
import tensorflow as tf
from .TransformerModel import Transformer
from .TransformerModel import create_masks
from .utils import CustomSchedule
#from .utils import loss_function
import os

from django.conf import settings

import json

#with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'NMT_PSL','config.json')) as f:
#    config = json.load(f)


#from configparser import ConfigParser
#config_file = 'ccc.ini'
#config = ConfigParser()
#config.read(config_file)
#print(config.sections())
#print(config['transformer'])


checkpoint_path = os.path.join(settings.BASE_DIR,"translator","nmt_model","checkpoints")
config_path = os.path.join(settings.BASE_DIR,"translator","nmt_model","config.json")


with open(config_path) as f:
    config = json.load(f)

start_tok = config['preprocessing']['start_tok']
end_tok = config['preprocessing']['end_tok']
vs = 10000
num_layers = config['transformer']['num_layers']
d_model = config['transformer']['d_model']
num_heads = config['transformer']['num_heads']
dff = config['transformer']['dff']
dropout_rate = config['transformer']['dropout_rate']
train_emb = False
bert_enc = False


class Model():

    

    def __init__(self):
        self.lang = BPEmb(lang="en", vs=10000)
        self.embedding_matrix = tf.keras.initializers.Constant(self.lang.vectors)
        self.transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vs, vs, 
                          pe_input=vs, 
                          pe_target=vs,
                          rate=dropout_rate,
                          emd_matrix=self.embedding_matrix,
                          train_emb = train_emb,
                          bert = bert_enc
                         )


        learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                          optimizer=self.optimizer)

        #ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=1)

        #if a checkpoint exists, restore the latest checkpoint.
        #if ckpt_manager.latest_checkpoint:
        #ckpt.restore(ckpt_manager.latest_checkpoint)
        print(tf.train.latest_checkpoint(checkpoint_path))
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
        print ('Latest checkpoint restored!!')


    def sentence_encoder(self, lang, text):
        print([start_tok]+lang.encode_ids(text)+[end_tok])
        return [start_tok]+lang.encode_ids(text)+[end_tok]

    def sentence_decoder(self,lang, words):
        return lang.decode_ids(words)

    def evaluate(self, inp_sentence):
        start_token = start_tok
        end_token = end_tok
    
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = self.sentence_encoder(self.lang, inp_sentence)
        encoder_input = tf.expand_dims(inp_sentence, 0)
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [start_tok]
        output = tf.expand_dims(decoder_input, 0)
        print(output.shape)
        
        for i in range(100):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output, bert_enc)
            
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
            # return the result if the predicted_id is equal to the end token
            if predicted_id == end_tok:
                return tf.squeeze( output, axis=0), attention_weights
        
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        
        predicted_sentence = self.sentence_decoder(
            self.lang,[int(i) for i in result 
                        if i not in [start_tok,end_tok]])  

        #print('Input: {}'.format(sentence))
        #print('Predicted translation: {}'.format(predicted_sentence))

        return predicted_sentence



obj = Model()
obj.translate("i am going to school")

