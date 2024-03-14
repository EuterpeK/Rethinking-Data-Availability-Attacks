device='1'
datasets='cifar10'
method='name'
surrogate='resnet18'
data_path='../data/'
def_radius=8
adv_radius=4
randomness=20

CUDA_VISIBLE_DEVICES=${device} python3 generate.py \
    --arch ${surrogate} \
    --dataset ${datasets} \
    --train-steps 5000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 2000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --atk-pgd-radius 4 \
    --atk-pgd-steps 10 \
    --atk-pgd-step-size 0.8 \
    --atk-pgd-random-start \
    --samp-num 5 \
    --report-freq 1000 \
    --save-freq 10000 \
    --data-dir ${data_path} \
    --save-dir ./noise/random-${randomness} \
    --save-name ${surrogate} \
    --defender pgd \
    --weight 1.0 \
    --random ${randomness}


for adversarial in 1
do 
    for arch in 'resnet18' 'vgg16-bn' 'resnet50' 'densenet-121' 'wrn-34-10'
    do
        CUDA_VISIBLE_DEVICES=${device} python3 train.py \
            --arch ${arch} \
            --dataset ${datasets} \
            --clean 0 \
            --train-steps 40000 \
            --batch-size 128 \
            --optim sgd \
            --lr 0.1 \
            --lr-decay-rate 0.1 \
            --lr-decay-freq 40000 \
            --weight-decay 5e-4 \
            --momentum 0.9 \
            --pgd-radius 4 \
            --pgd-steps 10 \
            --pgd-step-size 0.8 \
            --pgd-random-start \
            --report-freq 1000 \
            --save-freq 100000 \
            --noise-path ./noise/random-${randomness}/${surrogate}-noise.pkl \
            --data-dir ${data_path} \
            --save-dir ./trained \
            --save-name ${arch} \
            --adversarial ${adversarial} \
            --random ${randomness}
    done
done

