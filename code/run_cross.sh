DEVICE=0
TRAIN_TIME=1

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

TRAIN_FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
TEST_FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
python train_cross.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE
python train_cross.py --model xlnet-base-cased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --train_file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt' --test_file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time $TRAIN_TIME --device $DEVICE --sequence --train_file './my_data/train/recommenders2_TRAIN_Aug/recommenders2_TRAIN_Aug.txt' --test_file './my_data/train/TTS2_TRAIN_Aug/TTS2_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --train_file './my_data/train/faceswap2_TRAIN_Aug/faceswap2_TRAIN_Aug.txt' --test_file './my_data/train/_TRAIN_Aug/Real-Time-Voice-Cloning_TRAIN_Aug.txt'
python train_cross.py --model models/seBERT --local_model --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --train_file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt' --test_file './my_data/train/EasyOCR1_TRAIN_Aug/EasyOCR1_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time $TRAIN_TIME --device $DEVICE --sequence --train_file './my_data/train/TTS2_TRAIN_Aug/TTS2_TRAIN_Aug.txt' --test_file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
:
