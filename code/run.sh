DEVICE=0
TRAIN_TIME=5

<<'COMMENT'
replace the repo name in TRAIN_FILE and TEST FILE
repo list: [ 
        spleeter1.txt, 
        EasyOCR1.txt, 
        streamlit1.txt, 
        faceswap2.txt,
        EasyOCR1_TRAIN_Aug.txt
        recommenders2_TRAIN_Aug.txt, 
        TTS2_TRAIN_Aug.txt, 
        pytorch-CycleGAN-and-pix2pix3_TRAIN_Aug.txt, 
        jetson-inference1_TRAIN_Aug.xlsx, 
        Real-Time-Voice-Cloning_TRAIN_Aug.txt,
    ]
COMMENT

FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model bert-base-uncased --embed none --train_time 1 --do_predict --device 1 --disablefinetune --file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
python train.py --model bert-base-uncased --embed none --train_time 1 --do_predict --device 1 --disablefinetune --file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt'
python train.py --model bert-base-uncased --embed none --train_time 1 --do_predict --device 0 --file  './my_data/train/jetson-inference0_TRAIN_Aug/jetson-inference0_TRAIN_Aug.txt'
python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
python train.py --model xlnet-base-cased --embed none  --train_time 1 --do_predict --device 0 --sequence --file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt'

python train.py --model bert-base-uncased --embed none --train_time 1 --do_predict --device 0 --file './my_data/train/deepfacelab1_TRAIN_Aug/deepfacelab1_TRAIN_Aug.txt'
python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model roberta-base --embed none --train_time 1 --do_predict --device 1  --file './my_data/train/recommenders3_TRAIN_Aug/recommenders3_TRAIN_Aug.txt'
python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model albert-base-v2 --embed none --train_time 1 --do_predict --device 0 --file './my_data/train/recommenders3_TRAIN_Aug/recommenders3_TRAIN_Aug.txt'
python train.py --model textcnn --embed glove --train_time 1 --do_predict --device 0 --file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
python train.py --model bilstm --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model rcnn --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model roberta-base --embed none --train_time 1 --do_predict --device 1  --file './my_data/train/EasyOCR1_TRAIN_Aug/EasyOCR1_TRAIN_Aug.txt'
python train.py --model roberta-base --embed none --train_time 1 --do_predict --device 1  --file './my_data/train/EasyOCR1_TRAIN_Aug/EasyOCR1_TRAIN_Aug.txt'
python train.py --model rcnn --embed none --train_time 1 --do_predict --device 0 --file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'