DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python server_rnn.py \
    --BASE_DIR=$DATA/cnews \
    --EMBEDDING_DIM=64 \
    --SEQ_LENGTH=600  \
    --NUM_CLASSES=10  \
    --VOCAB_SIZE=5000  \
    --NUM_LAYERS=2  \
    --HIDDEN_DIM=128  \
    --RNN='GRU'  \
    --DROPOUT_KEEP_PROB=0.8  \
    --LEARNING_RATE=1e-3  \
    --BATCH_SIZE=128  \
    --NUM_EPOCHS=10 \
    --PRINT_PER_BATCH=100  \
    --SAVE_PER_BATCH=10 \
    --MODEL=$OUTPUT/'cnews/checkpoints/textrnn'