DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python predict_cnn.py \
    --BASE_DIR=$DATA/cnews \
    --EMBEDDING_DIM=64  \
    --SEQ_LENGTH=600  \
    --NUM_CLASSES=10  \
    --NUM_FILTERS=256  \
    --KERNEL_SIZE=5  \
    --VOCAB_SIZE=5000  \
    --HIDDEN_DIM=128  \
    --DROPOUT_KEEP_PROB=0.5  \
    --LEARNING_RATE=1E-3  \
    --BATCH_SIZE=64  \
    --NUM_EPOCHS=2  \
    --PRINT_PER_BATCH=100  \
    --SAVE_PER_BATCH=10 \
    --MODEL=$OUTPUT/'cnews/checkpoints/textcnn' \
    --DO_TRAIN \
    --DO_TEST