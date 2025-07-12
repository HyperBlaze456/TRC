# 음성 토크나이저 알파
해당 모델은 EnCodec, SoundStream 등과 같이 순수 음성 파형을 가지고 작동합니다.

인코더와 디코더의 구조는 순수하게 1D convolution으로 쌓아 차원을 줄이는 방식으로 작동하며, 중간에 양자화 모듈을 넣어 제한된 코드북으로 동작하도록 합니다.

현재 인코더/디코더는 24khz나 48khz의 입력을 받아 최종적으로 50hz 단위로 줄이나, 실제로는 25hz까지 줄여도 될 것 같습니다.

양자화기는 현재 모델의 핵심으로, 2개의 서로 다른 코드북을 가진 RVQ 형태로 가져갑니다. 이때 첫 번째 코드북은 단순 VQ로 음소를 CTC 방식으로 예측하고, 
두번째 코드북은 BSQ를 활용하며 아마 음소 정보를 제외한 나머지 음성학 정보(acoustic features)를 가지도록 훈련될 것입니다(희망적으로 말이죠)


## 훈련 과정
순수 파형에서 동작하기 때문에, 반드시 GAN을 활용하게 됩니다. 단순 reconstruction loss(L1/L2, STFT L1) 만으로는 부족하고, 여러 논문들과 과거 연구들에서 영감을 받아
(EnCodec, Enhanced RVQGAN/Descript Audio Codec) 등의 Multi-Scale Discriminators, Multi Period waveform Discriminators, Multi-resolution discriminator, STFT discriminator 등을 활용하거나 아예 새로운 구조의 Discriminator 모델을 만들어야 할 지도 모르겠습니다.

현재 연구자는 전반적으로 Non-AR 기반 생성형 모델(VAE, Diffusion, GAN) 등에 다소 약합니다. 아직 대학 수학에서 다루는 베이지안 등의 내용을 몰라 핵심 손실 함수(ELBO 등)의 전개를 완벽히 이해하지 못하고 있습니다.
GAN은 특히 아직 경험이 거의 전무하므로, 모델을 구현할 때 여러 가지 것들에 대해 신경 써 주십시오(간단한 이론, 모델 구조의 이유, 학습 루프의 단계 등등)