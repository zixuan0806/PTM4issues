DEVICE=0
TRAIN_TIME=1


TRAIN_FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
TEST_FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
python train_cross.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE
python train_cross.py --model xlnet-base-cased --embed none --train_time 10 --do_predict --device 1 --sequence --train_file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt' --test_file './my_data/train/openpose0_TRAIN_Aug/openpose0_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time 20 --device 1 --sequence --train_file './my_data/train/recommenders2_TRAIN_Aug/recommenders2_TRAIN_Aug.txt' --test_file './my_data/train/TTS2_TRAIN_Aug/TTS2_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time 1 --do_predict --device 1 --sequence --train_file './my_data/train/faceswap2_TRAIN_Aug/faceswap2_TRAIN_Aug.txt' --test_file './my_data/train/_TRAIN_Aug/Real-Time-Voice-Cloning_TRAIN_Aug.txt'
python train_cross.py --model models/seBERT --local_model --embed none --train_time 1 --do_predict --device 0 --sequence --train_file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt' --test_file './my_data/train/EasyOCR1_TRAIN_Aug/EasyOCR1_TRAIN_Aug.txt'
python train_cross.py --model xlnet-base-cased --embed none --train_time 1 --device 1 --sequence --train_file './my_data/train/TTS2_TRAIN_Aug/TTS2_TRAIN_Aug.txt' --test_file './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
:
