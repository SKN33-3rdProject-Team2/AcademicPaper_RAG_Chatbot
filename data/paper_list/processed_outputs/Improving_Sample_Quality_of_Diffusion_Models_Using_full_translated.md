# 원제: Improving Sample Quality of Diffusion Models Using Self-Attention Guidance

[본문 전체 마크다운 번역 결과]

# 확산 모델의 자체 어텐션 유도를 통한 샘플 품질 향상

수성 홍, 규성 이, 우석 장, 승룡 김  
고려대학교, 서울, 대한민국  
{susung1999, jpl358, jws1997, seungryong_kim}@korea.ac.kr

(a) SAG가 없는 ADM [7](위)와 SAG를 적용한 ADM(아래)  
(b) SAG가 없는 Stable Diffusion [31](위)와 SAG를 적용한 Stable Diffusion(아래)

> **그림 1: 유도되지 않은 샘플(위)과 자체 어텐션으로 유도된 샘플(아래)의 정성적 비교.** 분류기 유도(CG) [7] 또는 분류기 없는 유도(CFG) [16]와 달리, 자체 어텐션 유도(SAG)는 클래스 레이블이나 텍스트 프롬프트와 같은 외부 조건이나 추가 학습을 반드시 필요로 하지 않는다. SAG는 (a) 비조건부 ADM [7]과 (b) 빈 프롬프트를 사용한 Stable Diffusion [31]과 같은 사전 학습된 확산 모델이 생성하는 이미지의 세부 묘사를 향상한다.

## 초록

잡음 제거 확산 모델(denoising diffusion model, DDM)은 뛰어난 생성 품질과 다양성으로 주목받고 있다. 이러한 성공은 주로 분류기 유도 및 분류기 없는 유도와 같은 클래스 또는 텍스트 조건부 확산 유도 방법의 사용에 기인한다. 본 논문에서는 기존의 전통적인 유도 방법을 넘어서는 더욱 포괄적인 관점을 제시한다. 이러한 일반화된 관점에서, 생성 이미지의 품질을 향상하기 위한 새로운 무조건 및 무학습 전략을 소개한다.

간단한 해법으로서 블러 유도는 중간 샘플이 미세한 정보와 구조를 표현하는 데 더 적합하도록 만들어, 확산 모델이 적절한 유도 강도에서 더 높은 품질의 샘플을 생성할 수 있게 한다. 이를 개선한 자체 어텐션 유도(Self-Attention Guidance, SAG)는 확산 모델의 중간 자체 어텐션 맵을 활용하여 유도의 안정성과 효과를 높인다. 구체적으로 SAG는 각 반복 단계에서 확산 모델이 주목하는 영역만 적대적으로 블러 처리하고, 그에 따라 모델을 유도한다.

실험 결과, SAG는 ADM, IDDPM, Stable Diffusion, DiT를 포함한 다양한 확산 모델의 성능을 향상하는 것으로 나타났다. 또한 SAG를 기존의 유도 방법과 결합하면 성능이 추가로 향상된다.

## 1 서론

최근 잡음 제거 확산 모델(DDM) [35, 37, 14, 7, 15, 31]은 반복적인 잡음 제거 과정을 통해 잡음으로부터 이미지를 합성하며, 고품질과 높은 다양성을 동시에 갖춘 이미지 생성 성능으로 활발히 연구되고 있다. 이러한 뛰어난 성능의 배경에는 확산 유도 방법 [7, 23, 16]의 도입이 있다.

여러 연구는 확산 모델이 생성하는 이미지 샘플의 품질을 향상하려면 클래스 레이블 [7, 16]이나 캡션 [23]을 이용한 유도 기법이 필수적이라고 보고하였다. 그러나 이러한 유도 방법은 상당한 성능 향상에도 불구하고 외부 조건의 사용이라는 제약을 가진다. 예를 들어 분류기 유도(classifier guidance, CG) [7]는 별도의 분류기 학습을 요구하며, 분류기 없는 유도(classifier-free guidance, CFG) [16]는 레이블 제거를 통해 학습 과정을 복잡하게 만든다. 또한 두 방법 모두 어렵게 확보한 외부 조건에 의존하므로 조건부 설정에 국한된다.

이러한 한계를 고려하여, 본 연구에서는 확산 모델의 중간 샘플에 포함된 정보를 활용할 수 있는 보다 일반적인 확산 유도 공식을 제시한다. 이 공식은 기존 접근법 [7, 16, 23]에서 필요한 외부 정보라는 조건을 확산 유도에서 분리하고, 확산 모델을 유연하고 무조건적으로 유도할 수 있도록 한다. 이에 따라 외부 조건이 있거나 없는 경우 모두에 확산 유도를 적용할 수 있다.

먼저 중간 샘플에 포함된 모든 내부 정보가 유도로 활용될 수 있다는 일반화된 공식과 직관에 기반하여, 샘플 품질을 향상하기 위한 간단한 해법으로 블러 유도를 제안한다. 블러 유도는 가우시안 블러로 제거된 정보를 이용하여 중간 샘플을 유도한다. 가우시안 블러가 미세한 세부 정보를 자연스럽게 제거한다는 유리한 특성 [17, 20, 30]을 활용하는 것이다. 실험 결과, 이 방법은 적절한 유도 강도에서 샘플 품질을 향상한다. 그러나 유도 강도가 크면 전체 영역에 구조적 모호성이 도입될 수 있어 문제가 발생한다. 이 경우 열화된 입력에 대한 예측과 원본 입력에 대한 예측을 정렬하기 어려워진다.

큰 유도 강도에서도 블러 유도의 효과와 안정성을 높이기 위해, 확산 모델의 자체 어텐션 메커니즘을 탐구한다. 일반적으로 최근의 확산 모델 [14, 7, 24, 31, 15, 27]은 구조 내부에 자체 어텐션 모듈 [40, 8]을 포함한다. 자체 어텐션이 생성 과정에서 중요한 정보를 포착하는 핵심 요소라는 관찰 [18, 45, 46, 12]에 근거하여, 자체 어텐션 유도(SAG)를 제안한다. SAG는 확산 모델의 자체 어텐션 맵을 이용해 중요한 정보를 포함하는 영역을 적대적으로 블러 처리하고, 잔여 정보를 통해 확산 모델을 유도한다. 확산 모델의 역과정에서 어텐션 맵을 활용함으로써 외부 정보나 추가 학습 없이 자체 조건화를 통해 이미지 품질을 향상하고 아티팩트를 줄일 수 있다(그림 1).

본 연구의 기여는 다음과 같다.

- 조건부 유도 방법 [7, 16, 23]을 외부 조건 없이 모든 확산 모델에 적용할 수 있는 무조건 방법으로 일반화하여 유도의 적용 범위를 확장한다.
- 확산 모델의 내부 자체 어텐션 맵을 활용하는 새로운 유도 방법인 자체 어텐션 유도(SAG)를 제안한다. SAG는 외부 조건이나 추가 미세 조정 없이 샘플 품질을 향상한다.
- SAG가 기존의 조건부 모델 및 방법과 직교적인 관계임을 보이고, 다른 방법과 유연하게 결합하여 더 높은 성능을 얻을 수 있음을 입증한다.
- 제안 방법의 설계 선택을 정당화하고 효과를 검증하기 위한 광범위한 절제 연구를 수행한다.

프로젝트 페이지와 코드는 다음에서 확인할 수 있다.  
https://ku-cvlab.github.io/Self-Attention-Guidance/

## 2 관련 연구

### 잡음 제거 확산 모델

확산 모델 [35]은 점수 기반 모델 [37, 38]과 밀접한 관련이 있으며, 우수한 샘플링 품질과 다양성으로 큰 주목을 받아왔다. 선구적인 연구인 DDPM [14]은 이미지를 점진적으로 잡음 제거하는 반복 과정을 통해 복원한다. 이후 샘플링 과정의 품질과 속도를 개선하기 위한 여러 방법이 제안되었다 [36, 24, 31, 15, 7].

특히 IDDPM [24]은 확산 모델 역과정의 분산을 추가로 예측한다. DDIM [36]은 비마르코프 확산 과정을 도입하여 샘플링 속도를 높인다. LDM [31]은 잠재 공간에서 확산 과정을 처리하여 계산 비용을 줄인다.

큰 유도 강도에서 블러 유도의 효과와 안정성을 높이기 위해 확산 모델의 자체 어텐션 메커니즘을 활용한다. 최근의 확산 모델 [14, 7, 24, 31, 15, 27]은 일반적으로 U-Net 구조 내부의 중간 계층에 자체 어텐션 [40, 8]을 포함한다. SAG는 확산 모델의 자체 어텐션 맵을 이용하여 중요한 정보가 포함된 영역을 적대적으로 블러 처리하고, 잔여 정보를 통해 모델을 유도한다. 이 방법은 외부 정보나 추가 학습 없이 역과정 중 어텐션 맵을 활용하여 자체 조건화를 수행하며, 샘플 품질을 높이고 아티팩트를 줄인다. 의사 코드와 전체 파이프라인은 알고리즘 1과 그림 2(b)에 제시되어 있다.

### 확산 모델을 위한 샘플링 유도

최근에는 더 높은 품질의 이미지를 생성하기 위해 클래스 레이블에 기반한 확산 유도 방법이 제안되었다 [7, 16]. 분류기 유도(CG) [7]는 학습된 분류기를 사용하여 역과정을 특정 클래스 분포 방향으로 유도한다. 별도의 분류기를 사용하지 않는 대안으로 Ho와 Salimans [16]은 분류기 없는 유도(CFG)를 제안하였다. 구현이 간단하고 효과적이기 때문에 CFG는 다양한 고품질 확산 모델에 사용되고 있다 [29, 31, 39, 41, 23, 33]. Nichol 등 [23]은 이러한 유도 방법의 개념을 차용하여 CLIP [28] 유도와 CFG를 이용한 텍스트-이미지 생성을 제안하였다.

그러나 이러한 접근법은 레이블이 없는 데이터셋에 적용할 수 없고 추가 학습 절차가 필요하다는 한계가 있다 [7, 16]. 또한 외부 클래스 또는 텍스트 조건과 같이 어렵게 확보한 레이블을 요구하므로 조건부 확산 모델에 국한된다. CFG는 학습 과정에서 클래스 임베딩을 간헐적으로 0으로 만드는 추가 학습 과정도 필요하다 [16].

### 생성 모델의 자체 어텐션

자체 어텐션 메커니즘은 트랜스포머 기반 모델 [40]의 핵심 구성 요소이다. 전역 문맥을 인코딩하는 표현력과 능력으로 인해 자연어 처리 [40]에서 사실상의 표준 방법이 되었으며, 이는 컴퓨터 비전 분야에서도 자체 어텐션을 활용하도록 प्रेर했다 [8, 18, 45, 46].

Jiang 등 [18]과 Zhang 등 [45, 46]은 더 나은 이미지 품질을 위해 생성적 적대 신경망(GAN)에 자체 어텐션을 도입하였다. 이후 확산 모델 역시 구조에 자체 어텐션을 포함하였다. DDPM [14]은 U-Net [32]의 저해상도 계층에 자체 어텐션 층을 도입하여 이러한 흐름을 시작했다. Dhariwal과 Nichol [7]은 자체 어텐션 헤드 수와 해상도에 따른 성능 향상을 분석하였다. DiT [27]은 트랜스포머 기반 백본을 활용하여 높은 성능을 달성하였다.

## 3 사전 지식

### 잡음 제거 확률 모델

DDPM [14]은 반복적인 잡음 제거 과정을 통해 백색 잡음에서 이미지를 복원하는 모델이다. 이미지 $\mathbf{x}_0$와 시점 $t\in\{T,T-1,\ldots,1\}$에서의 분산 스케줄 $\beta_t$가 주어지면, 마르코프 과정으로 정의되는 순방향 과정을 통해 $\mathbf{x}_t$를 얻을 수 있다. 마찬가지로 $\epsilon_\theta(\mathbf{x}_t,t)$와 $\Sigma_\theta(\mathbf{x}_t,t)$로 매개화된 학습된 확산 모델이 주어지면 역과정을 정의할 수 있다. 여기서는 분산을 예측할 수도 있지만 [24, 7], $\Sigma_\theta(\mathbf{x}_t,t)=\sigma_t^2=\beta_t$로 설정한다 [14].

구체적으로 $\mathbf{x}_T\sim\mathcal{N}(0,\mathbf{I})$와 $\Sigma_\theta(\mathbf{x}_t,t)$가 주어졌을 때 DDPM은 다음을 계산하여 $\mathbf{x}_{T-1},\mathbf{x}_{T-2},\ldots,\mathbf{x}_0$를 샘플링한다.

$$\mathbf{x}_{t-1}=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t)\right)+\sigma_t\mathbf{z},\tag{1}$$

여기서 $\alpha_t=1-\beta_t$, $\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i$, $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$이며 $\epsilon_\theta$는 매개변수 $\theta$로 매개화된 신경망이다. 이후에는 간단히 $\epsilon_\theta(\mathbf{x}_t):=\epsilon_\theta(\mathbf{x}_t,t)$로 표기한다.

재매개화 기법을 이용하면 다음 식을 통해 시점 $t$에서 $\mathbf{x}_0$를 중간 복원한 $\hat{\mathbf{x}}_0$를 얻을 수 있다.

$$\hat{\mathbf{x}}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}.\tag{2}$$

### 분류기 유도와 분류기 없는 유도

GAN의 다양성과 충실도 사이의 상충 관계를 확산 모델에 도입하기 위해 Dhariwal과 Nichol [7]은 클래스 레이블 $c$를 사용하는 추가 분류기 $p(c|\mathbf{x}_t)$ 기반의 분류기 유도를 제안하였다. 유도 강도 $s>0$일 때 다음과 같이 표현된다.

$$\tilde{\epsilon}(\mathbf{x}_t,c)=\epsilon_\theta(\mathbf{x}_t,c)-s\sigma_t\nabla_{\mathbf{x}_t}\log p(c|\mathbf{x}_t).\tag{3}$$

여기서 $\epsilon_\theta(\mathbf{x}_t,c)$는 조건부 확산 모델의 출력이고, $\tilde{\epsilon}(\mathbf{x}_t,c)$는 분류기에 의해 유도된 출력이다.

Ho와 Salimans [16]은 추가 분류기 없이도 분류기 유도와 유사한 효과를 내는 분류기 없는 유도를 제안하였다.

$$\tilde{\epsilon}(\mathbf{x}_t,c)=\epsilon_\theta(\mathbf{x}_t,c)+s\left(\epsilon_\theta(\mathbf{x}_t,c)-\epsilon_\theta(\mathbf{x}_t)\right)=\epsilon_\theta(\mathbf{x}_t)+(1+s)\left(\epsilon_\theta(\mathbf{x}_t,c)-\epsilon_\theta(\mathbf{x}_t)\right).\tag{5}$$

## 4 확산 유도의 일반화

분류기 유도와 분류기 없는 유도는 조건부 확산 모델의 생성에 크게 기여했지만 [7, 16, 23], 외부 입력에 의존한다. 본 연구에서는 이를 외부 입력이 있는 경우와 없는 경우 모두로 확장한다. 또한 이 절의 마지막에서 CFG [16]를 본 프레임워크에 통합하는 방법을 보인다.

시점 $t$에서 확산 모델의 전체 입력은 일반화된 조건 $h_t$와 $h_t$가 제거된 섭동 샘플 $\bar{\mathbf{x}}_t$로 구성된다. 조건 $h_t$는 $\mathbf{x}_t$ 내부의 정보, 외부 조건 또는 이 둘 모두를 포함할 수 있다. 이 정의에 따라, $\bar{\mathbf{x}}_t$가 주어졌을 때 $h_t$를 예측한다고 가정하는 가상 회귀기 $p_{\mathrm{im}}(h_t|\bar{\mathbf{x}}_t)$를 사용하여 유도를 정의할 수 있다. 기존 연구 [38, 7]의 유도를 수정하면 다음을 얻는다.

$$\tilde{\epsilon}(\bar{\mathbf{x}}_t,h_t)=\epsilon_\theta(\bar{\mathbf{x}}_t,h_t)-s\sigma_t\nabla_{\bar{\mathbf{x}}_t}\log p_{\mathrm{im}}(h_t|\bar{\mathbf{x}}_t).\tag{8}$$

베이즈 규칙 $p_{\mathrm{im}}(h|\bar{\mathbf{x}}_t)\propto p(\bar{\mathbf{x}}_t|h)/p(\bar{\mathbf{x}}_t)$를 적용하면, 가상 회귀기의 점수는 다음과 같이 유도된다.

$$\nabla_{\bar{\mathbf{x}}_t}\log p_{\mathrm{im}}(h_t|\bar{\mathbf{x}}_t)=-\frac{1}{\sigma_t}\left(\epsilon^*(\bar{\mathbf{x}}_t,h_t)-\epsilon^*(\bar{\mathbf{x}}_t)\right),\tag{9}$$

여기서 $\epsilon^*$는 해당 회귀기의 실제 점수를 나타낸다. 이 항을 식 (8)에 대입하면 다음을 얻는다.

$$\tilde{\epsilon}(\bar{\mathbf{x}}_t,h_t)=\epsilon_\theta(\bar{\mathbf{x}}_t,h_t)+s\left(\epsilon_\theta(\bar{\mathbf{x}}_t,h_t)-\epsilon_\theta(\bar{\mathbf{x}}_t)\right)=\epsilon_\theta(\bar{\mathbf{x}}_t)+(1+s)\left(\epsilon_\theta(\bar{\mathbf{x}}_t,h_t)-\epsilon_\theta(\bar{\mathbf{x}}_t)\right).\tag{12}$$

이 식은 $\bar{\mathbf{x}}_t$가 확산 모델 $\epsilon_\theta$가 정의하는 데이터 다양체 내부에 있어야 한다는 제약을 유도한다. CFG [16]는 $\bar{\mathbf{x}}_t=\mathbf{x}_t$, $h_t=c$로 설정하고, 가상 회귀기 $p_{\mathrm{im}}(h_t|\bar{\mathbf{x}}_t)$를 [16]의 암묵적 분류기로 환원한 식 (12)의 특수한 경우이다.

이 정식화를 이용하면, 입력으로 잡음 이미지 $\mathbf{x}_t$만 사용하고 외부 레이블을 사용하지 않는 비조건부 모델 [7, 16]에도 확산 유도를 정의할 수 있다. 즉, 역과정의 중간 샘플에 포함된 시각 정보로 자체 조건화를 수행한다. 이에 따라 비조건부 모델에 적합한 $h_t$와 $\bar{\mathbf{x}}_t$를 찾는 방법을 논의하고, 다음 절에서 구체적인 유도를 제안한다.

## 5 자체 어텐션 맵을 활용한 샘플 품질 향상

4절의 유도에 따르면, $\mathbf{x}_t$에 포함된 중요한 정보 $h_t$를 추출하여 확산 모델의 역과정을 유도할 수 있다. 이러한 관찰에 영감을 받아, 사전 학습된 확산 모델에서 $\bar{\mathbf{x}}_t$가 데이터 분포 밖으로 벗어나는 문제를 완화하면서 역과정에 중요한 정보를 효과적으로 제공하는 자체 어텐션 유도(SAG)를 제안한다.

먼저 SAG의 기본 형태인 블러 유도를 설명한 뒤, 자체 어텐션 유도를 소개한다.

### 5.1 확산 모델을 위한 블러 유도

가우시안 블러는 입력 신호 $\hat{\mathbf{x}}_0$를 가우시안 필터 $G_\sigma$와 합성곱하여 출력 $\tilde{\mathbf{x}}_0$를 생성하는 선형 필터링 기법이다.

$$\tilde{\mathbf{x}}_0=\hat{\mathbf{x}}_0*G_\sigma,$$

여기서 $*$는 합성곱 연산을 의미한다. 표준편차 $\sigma$가 증가할수록 가우시안 블러는 입력 신호의 미세한 세부 정보를 줄이고 이를 상수에 가까운 형태로 평활화한다 [30]. 그 결과 국소적으로 구별하기 어려운 신호가 만들어진다.

구체적으로 식 (2)의 중간 복원 $\hat{\mathbf{x}}_0$를 가우시안 필터 $G_\sigma$로 먼저 블러 처리한다. 이후 $\epsilon_\theta(\mathbf{x}_t)$로 다시 잡음을 추가하여 $\tilde{\mathbf{x}}_t$를 생성한다. 이 과정은 블러가 가우시안 잡음까지 줄이는 부작용을 피하고, 유도가 무작위 잡음이 아닌 중간 콘텐츠에 의존하도록 한다. 간결한 표기를 위해, 잠재 공간 확산 모델 [31]을 포함할 수 있도록 $\mathbf{x}_t$를 잡음이 추가된 이미지 또는 공간 잠재 변수로 표기한다.

$\tilde{\mathbf{x}}_0$와 $\hat{\mathbf{x}}_0$ 사이에는 정보 불균형이 존재한다. 즉, $\hat{\mathbf{x}}_0$는 더 많은 미세 정보를 포함한다. 이 통찰에 기반하여 식 (12)의 특수한 형태인 블러 유도를 정의한다. 블러 유도는 확산 과정에서 중간 복원으로부터 일부 정보를 의도적으로 제거하고, 제거된 정보를 이용해 해당 정보와 더 잘 부합하는 방향으로 예측을 유도한다.

블러 유도에서는 $\bar{\mathbf{x}}_t=\tilde{\mathbf{x}}_t$, $h_t=\mathbf{x}_t-\tilde{\mathbf{x}}_t$로 설정한다. 실제로 결합 입력 $(\tilde{\mathbf{x}}_t,h_t)$는 단순히 $\mathbf{x}_t=\tilde{\mathbf{x}}_t+h_t$로 계산된다. $\mathbf{x}_t-\tilde{\mathbf{x}}_t$는 블러 처리 전에 존재했던 정보를 보존하므로, 제거된 중요한 정보에 맞게 확산 과정을 유도한다.

가우시안 블러는 $\sigma$가 적절한 경우 신호가 원래 데이터 다양체에서 크게 벗어나지 않도록 하는 특성이 있다 [17, 20, 30]. 이미지는 본질적으로 어느 정도 블러를 포함하므로, 가우시안 블러는 사전 학습된 확산 모델에 적용하기에 특히 적합하다. 이는 공간 잠재 변수가 국소 구조와 같은 저수준 정보도 포함하는 잠재 확산 모델 [31]에도 해당한다.

표 5의 “Global” 결과에서 확인할 수 있듯이 블러 유도는 품질 지표 측면에서 기준선 성능을 향상한다. 그러나 큰 유도 강도($s>5.0$)에서는 그림 3의 위쪽 행과 같이 잡음이 많은 결과를 생성한다. 이는 전역 블러가 전체 영역에 구조적 모호성을 도입하기 때문이라고 추정한다. 열화된 입력에 대한 예측과 원본 입력에 대한 예측을 정렬하기 어려워지고, 이러한 오차가 여러 시점에 걸쳐 누적되어 잡음이 발생한다.

이 문제는 전역 블러보다 역과정에서 더 세밀하고 관련성 높은 정보를 포착하는 적응적 방법이 필요함을 보여준다. 이에 따라 확산 모델의 자체 어텐션 맵을 활용하는 SAG를 제안한다.

### 5.2 확산 모델을 위한 자체 어텐션 유도

자체 어텐션 메커니즘 [8, 40]은 확산 모델의 핵심 구성 요소로 알려져 있다 [7, 14]. 확산 모델의 백본에 구현된 자체 어텐션은 생성 과정에서 입력의 중요한 부분에 주목할 수 있게 한다 [18, 45, 46, 12].

그림 4는 이러한 정보 포착의 사례를 보여준다. ADM [7]에서 얻은 자체 어텐션 마스크의 영역은 최종 생성 이미지의 고주파 세부 정보와 겹친다. 확산 모델이 정교하게 표현해야 하는 세부 정보는 이미지 생성 [4, 42, 43]과 인간의 인지 [5]에 중요한 요소이다.

식 (7)의 자체 어텐션 맵을 통합하기 위해, 먼저 여러 자체 어텐션 맵 $A_t^S\in\mathbb{R}^{N\times(HW)\times(HW)}$에 전역 평균 풀링(GAP)을 적용하여 $\mathbb{R}^{HW}$ 차원으로 집계한다. 이후 이를 $\mathbb{R}^{H\times W}$로 재구성하고 최근접 이웃 업샘플링을 수행하여 $\mathbf{x}_t$와 동일한 해상도로 맞춘다.

$$A_t=\mathrm{Upsample}\left(\mathrm{Reshape}\left(\mathrm{GAP}(A_t^S)\right)\right).\tag{13}$$

실제로는 $A_t$의 평균값으로 설정한 마스킹 임계값 $\psi$를 사용하여 자체 어텐션 맵에 따라 $\mathbf{x}_t$의 마스크된 패치만 블러 처리한다.

$$M_t=\mathbf{1}(A_t>\psi),$$

$$\hat{\mathbf{x}}_t=(1-M_t)\odot\mathbf{x}_t+M_t\odot\tilde{\mathbf{x}}_t,\tag{15}$$

여기서 $\odot$는 아다마르 곱을 의미하며 $\tilde{\mathbf{x}}_t$는 5.1절과 동일한 방식으로 얻는다. 유도된 잡음 예측은 다음과 같다.

$$\tilde{\epsilon}(\mathbf{x}_t)=\epsilon_\theta(\hat{\mathbf{x}}_t)+(1+s)\left(\epsilon_\theta(\mathbf{x}_t)-\epsilon_\theta(\hat{\mathbf{x}}_t)\right).\tag{16}$$

식 (16) 역시 식 (12)의 특수한 경우이다. 이때 $h_t=M_t\odot\mathbf{x}_t-M_t\odot\tilde{\mathbf{x}}_t$, $\bar{\mathbf{x}}_t=\hat{\mathbf{x}}_t$이며 결합 입력은 5.1절과 같이 단순한 합으로 계산된다.

블러 유도와 달리 $\hat{\mathbf{x}}_t$는 $\mathbf{x}_t$의 블러 처리되지 않은 패치를 명시적으로 포함한다. 따라서 큰 유도 강도에서도 $\epsilon_\theta(\hat{\mathbf{x}}_t)$가 원래 예측에서 지나치게 벗어나는 것을 방지하고(그림 3), 역과정에 중요한 정보를 적대적으로 효과적으로 은닉할 수 있다.

### 알고리즘 1 자체 어텐션 유도(SAG) 샘플링

- `Model(x_t)`: 입력 $x_t$를 받아 예측 잡음 $\epsilon_t$, 분산 $\Sigma_t$, 자체 어텐션 맵 $A_t$를 출력하는 확산 모델
- `Gaussian-Blur(\hat{x}_0)`: 가우시안 블러 함수

초기화: $\mathbf{x}_T\sim\mathcal{N}(0,\mathbf{I})$

$t=T,T-1,\ldots,1$에 대해 반복:

1. $(\epsilon_t,\Sigma_t,A_t)\leftarrow\mathrm{Model}(\mathbf{x}_t)$
2. $M_t\leftarrow\mathbf{1}(A_t>\psi)$
3. $\hat{\mathbf{x}}_0\leftarrow(\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_t)/\sqrt{\bar{\alpha}_t}$
4. $\tilde{\mathbf{x}}_0\leftarrow\mathrm{Gaussian\text{-}Blur}(\hat{\mathbf{x}}_0)$
5. $\tilde{\mathbf{x}}_t\leftarrow\sqrt{\bar{\alpha}_t}\tilde{\mathbf{x}}_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t$
6. $\hat{\mathbf{x}}_t\leftarrow(1-M_t)\odot\mathbf{x}_t+M_t\odot\tilde{\mathbf{x}}_t$
7. $\hat{\epsilon}_t\leftarrow\mathrm{Model}(\hat{\mathbf{x}}_t)$
8. $\tilde{\epsilon}_t\leftarrow\hat{\epsilon}_t+(1+s)(\epsilon_t-\hat{\epsilon}_t)$
9. $\mathbf{x}_{t-1}\sim\mathcal{N}\left(\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon}_t\right),\Sigma_t\right)$

반환: $\mathbf{x}_0$

> **그림 2: 분류기 없는 유도 [16]와 자체 어텐션 유도(SAG)의 비교.** 분류기 없는 유도는 외부 클래스 정보를 사용하는 반면, SAG는 자체 어텐션을 통해 내부 정보를 추출하여 모델을 유도한다. 따라서 SAG는 학습과 조건이 필요 없다.

## 6 실험

### 6.1 실험 설정

실험에서는 각각 NVIDIA GeForce RTX 3090 GPU 8개가 장착된 서버 두 대를 사용하여 샘플을 생성하였다. ADM [7], IDDPM [24], Stable Diffusion [31], DiT [27]의 사전 학습 모델을 기반으로 하였으며, 모든 가중치는 공개 저장소에서 사용하였다. [7]과 동일한 평가 지표인 FID [13], sFID [22], IS [34], 개선된 정밀도 및 재현율 [19]을 사용하였다.

### 6.2 실험 결과

#### SAG를 이용한 비조건부 생성

CG와 CFG에는 없는 조건 독립성을 보이기 위해 비조건부 모델에서 SAG의 효과를 평가하였다. ADM을 ImageNet [6] 256×256, LSUN Cat [44], LSUN Horse [44]에서 평가하였다.

> **표 1: 256×256 이미지로 사전 학습된 ADM [7]에서 자체 어텐션 유도를 적용한 50K 샘플 결과. 최상의 값은 굵게 표시하였다.**

| 데이터셋 | 입력 | 단계 수 | SAG | FID(↓) | sFID(↓) | IS(↑) | 정밀도(↑) | 재현율(↑) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| ImageNet 256×256 | 비조건부 | - | ✗ | 26.21 | 6.35 | 39.70 | 0.61 | 0.63 |
| ImageNet 256×256 | 비조건부 | - | ✓ | **20.08** | **5.77** | **45.56** | **0.68** | 0.59 |
| ImageNet 256×256 | 조건부 | - | ✗ | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| ImageNet 256×256 | 조건부 | - | ✓ | **9.41** | **5.28** | **104.79** | **0.70** | 0.62 |
| LSUN Cat 256×256 | 비조건부 | - | ✗ | 7.03 | 8.24 | - | 0.60 | 0.53 |
| LSUN Cat 256×256 | 비조건부 | - | ✓ | **6.87** | **8.21** | - | 0.60 | 0.50 |
| LSUN Horse 256×256 | 비조건부 | - | ✗ | 3.45 | 7.55 | - | 0.68 | 0.56 |
| LSUN Horse 256×256 | 비조건부 | - | ✓ | **3.43** | **7.51** | - | 0.68 | 0.55 |

SAG는 비조건부 모델에서 FID, sFID, IS를 일관되게 향상하지만 재현율은 낮춘다. 최근 연구 [7, 16]에서 설명한 것처럼 이는 샘플 충실도와 다양성 사이의 상충 관계 때문으로 추정된다. 그림 6의 동일한 무작위 시드를 사용한 비교에서 확인할 수 있듯이, 내부 조건의 자체 조건화로 인해 정성적 품질은 향상된다.

IDDPM [24]의 비조건부 모델에도 SAG를 적용하였다. ImageNet 64×64로 학습된 모델에서 FID가 19.2에서 18.0으로 향상되었다.

> **표 2: ImageNet 64×64로 사전 학습된 IDDPM [24]에서 자체 어텐션 유도를 적용한 50K 결과.**

| 스케줄 | 목적 함수 | 입력 | SAG | FID(↓) |
|---|---|---|---|---:|
| cosine | $L_{\mathrm{hybrid}}$ | 비조건부 | ✗ | 19.2 |
| cosine | $L_{\mathrm{hybrid}}$ | 비조건부 | ✓ | **18.0** |

Stable Diffusion에서도 빈 프롬프트를 사용하여 SAG를 평가하였다. 동일한 무작위 시드로 SAG 적용 여부만 달리한 500쌍의 이미지를 대상으로 인간 평가를 수행한 결과, SAG를 사용한 샘플이 사람에게 더 시각적으로 선호되거나 현실적으로 평가되었다(그림 5, 그림 7).

또한 Stable Diffusion에서 CFG와 SAG를 결합하여 텍스트-이미지 생성으로 범위를 확장하였다. SAG를 적용한 샘플은 자체 조건화 효과로 인해 더 높은 품질과 더 적은 아티팩트를 보였다. 빈 프롬프트를 사용한 경우에도 품질이 뚜렷하게 향상되어, SAG가 외부 조건에 의존하지 않음을 확인하였다.

#### SAG를 이용한 조건부 생성

식 (12)는 조건에 구애받지 않으므로 SAG는 조건부 모델에도 적용할 수 있다. ImageNet 256×256으로 조건부 학습된 ADM에서 실험한 결과, 비조건부 모델과 유사한 성능 향상이 나타났다.

#### CG 및 CFG와의 직교성

SAG는 외부 조건을 사용하는 CG [7]와 결합할 수 있다. ImageNet 128×128 모델에서 네 가지 조합을 비교하였다.

> **표 3: CG [7]와 SAG의 호환성. 결과는 ImageNet 128×128으로 학습된 ADM에서 얻었다.**

| CG | SAG | FID(↓) | sFID(↓) | 정밀도(↑) | 재현율(↑) |
|---|---|---:|---:|---:|---:|
| ✗ | ✗ | 5.91 | 5.09 | 0.70 | 0.65 |
| ✓ | ✗ | 2.97 | 5.09 | 0.78 | 0.59 |
| ✗ | ✓ | 5.11 | **4.09** | 0.72 | 0.65 |
| ✓ | ✓ | **2.58** | 4.35 | **0.79** | 0.59 |

두 방법을 함께 사용하면 FID와 정밀도가 추가로 향상된다. 이는 SAG가 기존 유도 방법과 직교적인 요소를 가지며 동시에 사용할 수 있음을 의미한다.

CFG [16]와의 결합도 평가하였다. 자체 어텐션 층을 포함하는 트랜스포머 기반 DiT-XL/2 [27]를 사용하였다.

> **표 4: CFG [16]와 SAG의 호환성. 결과는 ImageNet 256×256으로 학습된 DiT-XL/2에서 얻었다.**

| 모델 | CFG | SAG | FID(↓) |
|---|---|---|---:|
| DiT-XL/2 [27] | ✓ | ✗ | 2.27 |
| DiT-XL/2 [27] | ✓ | ✓ | **2.16** |

### 6.3 절제 연구 및 분석

#### 마스킹 전략

ADM에서 10K 샘플을 사용하여 다양한 마스킹 전략을 비교하였다. 다른 전략에서는 SAG의 임계값이 1.0일 때 마스크되는 영역과 동일하게 이미지의 40%를 마스크하였다.

> **표 5: 마스킹 전략에 대한 절제 연구. ImageNet 128×128으로 학습된 ADM에서 얻은 결과.**

| 마스킹 전략 | FID(↓) | IS(↑) |
|---|---:|---:|
| 기준선 | 5.98 | 141.72 |
| 전역(5.1절의 블러 유도) | 5.82 | 143.15 |
| 고주파 | 5.74 | 148.87 |
| 무작위 | 5.68 | 148.99 |
| 정사각형 | 5.68 | 146.50 |
| 자체 어텐션(SAG) | **5.47** | **151.12** |
| DINO [3] 어텐션 | 5.63 | 146.18 |

자체 어텐션 마스킹이 다른 전략보다 우수했다. 특히 전역 마스킹, 즉 블러 유도는 가장 낮은 성능을 보여 SAG의 동기를 뒷받침하였다. $\hat{\mathbf{x}}_0$에 FFT 기반 고주파 마스크와 DINO [3]의 자체 어텐션 마스크를 적용했지만, FID와 IS 모두에서 SAG보다 낮은 성능을 보였다.

#### 가우시안 블러의 $\sigma$

$\sigma\in\{1,3,9,27\}$ 및 극단적인 경우를 포함하여 10K 샘플로 평가하였다. $\sigma\to\infty$이면 필터는 신호 콘텐츠를 점진적으로 블러 처리하여 모든 픽셀을 평균값으로 만든다. 반대로 $\sigma\to0$이면 신호는 변하지 않는다.

> **표 6: 가우시안 블러의 $\sigma$에 대한 절제 연구.**

|  | 기준선($\sigma\to0$) | $\sigma=1$ | $\sigma=3$ | $\sigma=9$ | $\sigma=27$ | 평균 픽셀($\sigma\to\infty$) |
|---|---:|---:|---:|---:|---:|---:|
| FID(↓) | 5.98 | 5.58 | **5.47** | 5.70 | 5.80 | 5.84 |
| IS(↑) | 141.72 | 145.85 | **151.12** | 148.70 | 147.83 | 147.52 |

SAG는 $\sigma$의 변화에 강건하지만 최상의 성능을 내는 최적의 $\sigma$가 존재한다. 최적값은 입력 해상도에도 의존하며, 일반적으로 입력 해상도가 높을수록 더 큰 $\sigma$가 필요하다.

#### 유도 강도

ADM에서 유도 강도를 변화시키며 10K 샘플을 평가하였다. 유도 강도 $s=-0.1,0.1,0.2,0.3,0.4$를 시험한 결과, FID, sFID, IS는 $s=0.1$에서 최상이었다. 정밀도는 $s=0.3$에서 가장 높았다. 음의 유도 강도($s=-0.1$)나 지나치게 큰 유도 강도($s\geq0.4$)는 샘플 품질을 저하시켰다.

#### 계산 비용

> **표 7: 계산 비용.**

| 방법 | GPU 메모리 | 실행 시간 |
|---|---:|---:|
| 유도 없음 | 12,167 MB | 108.27초 |
| SAG | 12,209 MB | 186.60초 |
| CFG [16] | 12,218 MB | 190.27초 |

SAG의 메모리와 시간 사용량은 CFG와 거의 동일하다. 블러와 마스킹 등 SAG 연산으로 인한 추가 부담은 무시할 수 있다. 다만 추가적인 순전파 단계가 필요하므로 유도하지 않는 경우보다 비용이 높다.

## 7 결론

본 연구에서는 확산 모델 내부의 정보를 활용하여 고품질 이미지를 합성하는 일반적인 유도 공식을 제시하였다. 제안한 자체 어텐션 유도는 조건과 학습이 필요 없으며, ADM, IDDPM, Stable Diffusion, DiT 등 다양한 확산 모델에 적용할 수 있다. 또한 자체 조건화를 통해 이미지 품질을 향상하고 아티팩트를 줄인다.

실험 결과는 제안 방법의 효과와 자체 어텐션 유도가 기존 유도 방법과 직교적이라는 점을 입증하였다. 본 연구의 발견과 유도 방법의 일반화가 잡음 제거 확산 모델 및 그 유도 기법에 대한 후속 연구의 새로운 방향을 열기를 기대한다.

## 감사의 글

본 연구는 대한민국 과학기술정보통신부의 지원(IITP-2022-2020-0-01819, ICT Creative Consilience 프로그램)과 한국연구재단의 지원(NRF-2021R1C1C1006897)을 받았다. 또한 삼성전자 MX사업부의 지원을 받았다.

## 부록

본 부록에서는 DDPM [7]에 대한 추가 설명, 제안 방법의 구현 세부 사항, 추가 분석 및 결과, 인간 평가 프로토콜을 제시한다. 마지막으로 한계와 향후 연구 방향을 논의한다.

### A. 잡음 제거 확률 모델

DDPM [14]은 반복적인 잡음 제거 단계를 통해 백색 잡음으로부터 이미지를 생성하는 생성 모델이다. 이미지 $\mathbf{x}_0$와 임의의 시점 $t\in\{1,2,\ldots,T\}$에 대한 분산 스케줄 $\beta_t$가 주어지면, DDPM의 순방향 과정은 다음과 같은 마르코프 과정으로 정의된다.

$$q(\mathbf{x}_{t+1}|\mathbf{x}_t)=\mathcal{N}\left(\mathbf{x}_{t+1};\sqrt{1-\beta_t}\mathbf{x}_t,\beta_t\mathbf{I}\right).\tag{17}$$

$\mathbf{x}_0$에서 $\mathbf{x}_t$를 닫힌 형태로 직접 얻을 수도 있다.

$$q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}\left(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I}\right),\tag{18}$$

여기서 $\alpha_t=1-\beta_t$, $\bar{\alpha}_t=\prod_{i=1}^{t}\alpha_i$이다.

역과정은 다음과 같이 정의된다.

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}\left(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t,t),\Sigma_\theta(\mathbf{x}_t,t)\mathbf{I}\right),\tag{19}$$

여기서 $\mu_\theta$와 $\Sigma_\theta$는 매개변수 $\theta$를 갖는 신경망이다. 학습 과정에서는 DDPM과 같이 $\Sigma_\theta$를 상수 $\sigma_t^2=\beta_t$로 고정하고, $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$를 다음의 순방향 사후분포와 비교한다.

$$q(\mathbf{x}_{t-1}|\mathbf{x}_0,\mathbf{x}_t)=\mathcal{N}\left(\mathbf{x}_{t-1};\tilde{\mu}_t(\mathbf{x}_0,\mathbf{x}_t),\tilde{\beta}_t\mathbf{I}\right).\tag{20}$$

Ho 등 [14]은 $\mu_\theta$와 $\tilde{\mu}_t$를 직접 비교하는 대신, 재매개화 후 다음의 단순화된 목적 함수로 $\epsilon_\theta$를 최적화하는 것이 유리함을 보였다.

$$\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\quad\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I}).\tag{21}$$

$$L_{\mathrm{simple}}=\mathbb{E}_{\mathbf{x}_0,t,\boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon}-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},t\right)\right\|^2\right].\tag{22}$$

$\mathbf{x}_{t-1}\sim p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$를 샘플링할 때에는 다음 식을 이용하여 $\mathbf{x}_T$에서 $\mathbf{x}_0$까지 계산한다.

$$\mathbf{x}_{t-1}=\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t,t)\right)+\sigma_t\mathbf{z},\tag{23}$$

여기서 $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$이다. 식 (21)을 다시 쓰면 각 시점에서 $\mathbf{x}_0$의 예측값 $\hat{\mathbf{x}}_0$를 다음과 같이 얻을 수 있다.

$$\hat{\mathbf{x}}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}.\tag{24}$$

### B. 추가 구현 세부 사항

#### B.1 환경 설정

ADM [7], IDDPM [24], Stable Diffusion v1.4 [31], DiT [27]의 사전 학습 모델을 사용하였다. 각 모델의 PyTorch [26] 구현을 기반으로 하였으며, 모든 가중치는 공개 저장소에서 가져왔다. 샘플링에는 NVIDIA GeForce RTX 3090 GPU 8개가 탑재된 서버 두 대를 사용하였다.

#### B.2 선택적 블러

5.2절의 선택적 블러는 다음과 같이 효율적으로 구현한다. 먼저 $\mathbf{x}_t$의 중간 복원 $\hat{\mathbf{x}}_0$를 블러 처리한다 [14]. 그런 다음 $\hat{\mathbf{x}}_0$와 블러 처리된 $\hat{\mathbf{x}}_0$에 각각 $1-M_t$와 $M_t$를 적용한다. 출력들을 결합한 후, 위에서 $\hat{\mathbf{x}}_0$를 계산할 때 사용한 예측 잡음 $\epsilon_\theta(\mathbf{x}_t)$를 다시 추가한다. 이 과정은 본문의 식 (15)와 동일한 $\hat{\mathbf{x}}_t$를 생성한다.

#### B.3 SAG와 CFG의 결합

Stable Diffusion [31]과 DiT [27]에서 SAG와 CFG [16]를 단순하게 결합하면 조건부 모델과 무조건부 모델 각각에 대해 SAG를 계산해야 하므로 네 번의 순전파가 필요하다. 실제로는 다음과 같이 유도된 잡음 예측을 효율적으로 계산할 수 있다.

$$\tilde{\epsilon}(\mathbf{x}_t)=\epsilon_\theta(\mathbf{x}_t,c)+s_c\left(\epsilon_\theta(\mathbf{x}_t,c)-\epsilon_\theta(\mathbf{x}_t)\right)+s_s\left(\epsilon_\theta(\mathbf{x}_t)-\epsilon_\theta(\bar{\mathbf{x}}_t)\right),\tag{25}$$

여기서 $s_c$와 $s_s$는 각각 CFG와 SAG의 유도 강도이며, $c$는 텍스트 프롬프트이다.

#### B.4 하이퍼파라미터 설정

> **표 8: 하이퍼파라미터 설정.**

| 모델 및 데이터셋 | 유도 강도 | 임계값 | 계층 | $\sigma$ |
|---|---:|---:|---|---:|
| ADM, ImageNet 256×256 비조건부 | 0.5, 0.8 | 1.0 | Output 2 | 9 |
| ADM, ImageNet 256×256 조건부 | 0.2 | 1.0 | Output 2 | 9 |
| ADM, LSUN Cat 256×256 | 0.05 | 1.0 | Output 2 | 9 |
| ADM, LSUN Horse 256×256 | 0.01 | 1.0 | Output 2 | 9 |
| ADM, ImageNet 128×128 | 0.1 | 1.0 | Output 8 | 3 |
| IDDPM, ImageNet 64×64 비조건부 | 0.05 | 1.0 | Output 7 | - |
| Stable Diffusion | 0.75, 1.0 | 1.0 | Middle | - |
| DiT | 0.005 | 1.0 | 13번째 블록 | - |

### C. 추가 분석 및 결과

#### C.1 확산 모델의 자체 어텐션 탐구

ADM [7]의 U-Net [32]에서 8×8, 16×16, 32×32 해상도의 자체 어텐션 맵을 시각화하였다. 중간 시점의 어텐션 맵은 생성 이미지의 구조를 포착한다. 또한 U-Net의 서로 다른 헤드와 계층에서 자체 어텐션 마스크를 추출하였다. “평균”은 네 개 헤드의 어텐션 맵을 평균한 후 얻은 마스크를 의미한다.

ADM의 자체 어텐션 마스크와 DINO [3]의 마스크를 비교하면, ADM의 마스크는 여러 객체와 확산 모델이 정교하게 표현해야 하는 고주파 세부 정보에 더 많이 주목한다. 이를 바탕으로 확산 모델 자체 어텐션이 샘플의 주파수와 의미 정보에 주목하는 두 측면을 분석하였다.

먼저 높은 어텐션 점수를 갖는 패치와 전체 패치의 주파수 스펙트럼을 비교하여 자체 어텐션 맵과 주파수의 상관관계를 조사하였다. 높은 어텐션 패치는 더 많은 고주파 세부 정보를 포함했다(그림 11). 이어서 자체 어텐션 맵이 전경 객체와 얼마나 일치하는지 평가한 결과, 모든 해상도에서 일정한 의미 정보를 포착하는 것으로 나타났다(표 9, 그림 12).

> **표 9: 자체 어텐션 마스크의 의미 분석.** $\psi$는 마스킹 임계값이며, `% Diff.`는 무작위 기준에 대한 IoU의 백분율 차이를 의미한다.

| 패치 크기 | $\psi$ | 무작위 | 자체 어텐션 | 차이 |
|---|---:|---:|---:|---:|
| 8×8 | 1.0 | 0.16 | 0.23 | +44% |
| 8×8 | 1.3 | 0.09 | 0.14 | +56% |
| 16×16 | 1.0 | 0.18 | 0.25 | +39% |
| 16×16 | 1.3 | 0.05 | 0.11 | +120% |
| 32×32 | 1.0 | 0.18 | 0.26 | +44% |
| 32×32 | 1.3 | 0.04 | 0.10 | +150% |

#### C.2 추가 절제 연구

자체 어텐션 마스킹 임계값이 블러 영역의 비율에 미치는 영향을 10K 샘플로 평가하였다. $\psi=0.7,1.0,1.3$을 시험한 결과 $\psi=1.0$에서 가장 높은 성능을 얻었다.

> **표 10: 마스킹 임계값 $\psi$에 대한 절제 연구.**

|  | 기준선 | $\psi=0.7$ | $\psi=1.0$ | $\psi=1.3$ |
|---|---:|---:|---:|---:|
| FID(↓) | 5.98 | 5.67 | **5.47** | 5.66 |
| IS(↑) | 141.72 | 148.60 | **151.12** | 145.58 |

어텐션 맵을 추출하는 계층의 영향도 10K 샘플로 평가하였다. 인코더와 디코더에서 각 해상도의 마지막 자체 어텐션 계층을 선택하고, 인코더와 디코더를 나누는 병목 계층도 포함하였다.

> **표 11: 어텐션 맵 추출 계층에 대한 절제 연구.**

| 계층 | 기준선 | In. 11 | In. 8 | Mid. | Out. 2 | Out. 5 | Out. 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FID(↓) | 5.98 | 5.54 | 5.61 | 5.63 | 5.59 | 5.57 | **5.47** |
| IS(↑) | 141.72 | 150.07 | 148.20 | 143.44 | 150.62 | 141.73 | **151.12** |

추출 계층과 관계없이 기준선보다 성능이 일관되게 향상되었으며, 최종 계층의 자체 어텐션을 사용했을 때 FID와 IS가 가장 우수했다.

#### C.3 정성적 결과

본문의 샘플 외에도 ImageNet 128×128으로 사전 학습된 ADM(그림 17), LSUN Cats(그림 18), LSUN Horse(그림 19)에서 SAG를 적용한 무작위 샘플을 제시한다.

## D. 인간 평가 프로토콜

Stable Diffusion [31]의 샘플을 사용하여 빈 프롬프트와 SAG 적용 여부를 달리한 500쌍을 생성하였다. SAG를 적용한 샘플의 SAG 강도는 1.0으로 설정하였다. 각 쌍은 비교를 위해 동일한 시드를 공유한다.

50명의 참가자에게 SAG 적용 샘플 4개와 미적용 샘플 4개로 구성된 두 그룹을 제시하고, 더 높은 이미지 품질을 갖는 그룹을 선택하도록 요청하였다. 질문의 예시는 그림 13에 제시되어 있다. 쌍을 임의로 선택하거나 필터링하지 않았으며, 응답에 대한 후처리도 수행하지 않았다.

> **그림 13: 평가 질문의 예시.** 참가자에게 어느 행이 제안 방법으로 생성되었는지는 알려주지 않는다.

## E. 한계 및 향후 연구

자체 조건화가 증가하면 일반적으로 사람이 보기에 더 매력적인 결과가 생성되지만, 생성 이미지의 다양성과 참신성이 감소할 가능성도 고려해야 한다. 다만 현재 단계에서는 유도 강도를 조절하여 SAG의 영향을 효과적으로 제어할 수 있으므로 유용한 응용이 가능하다.

또한 SAG는 순전파 단계를 두 배로 요구한다. 이는 CFG [16]에도 공통적인 문제이며 해결이 필요하다. 가능한 해법으로는 유도 정보를 확산 모델에 증류하는 방법 [21]이 있다. 이를 통해 품질 저하 없이 SAG와 CFG의 계산 비용을 줄일 수 있을 것이다.

자체 어텐션 기반 유도는 연속값으로 토큰 확률을 근사하는 대신 토큰 확률을 직접 모델링하는 이산 확산 모델 [39, 10]에 더 적합할 수도 있다. 이러한 모델과 제안 방법의 통합은 향후 연구의 흥미로운 주제이다.

## 참고문헌

참고문헌의 서지 정보는 원문의 인용 형식과 번호를 유지한다.

[1] Dmitry Baranchuk, Andrey Voynov, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. 확산 모델을 이용한 레이블 효율적 의미론적 분할. ICLR, 2021.

[2] Emmanuel Asiedu Brempong, Simon Kornblith, Ting Chen, et al. 의미론적 분할을 위한 잡음 제거 사전 학습. CVPR, 2022.

[3] Mathilde Caron, Hugo Touvron, Ishan Misra, et al. 자기 지도 비전 트랜스포머에서 나타나는 특성. ICCV, 2021.

[4] Yuanqi Chen, Ge Li, Cece Jin, Shan Liu, and Thomas Li. SSD-GAN: 공간 및 스펙트럼 영역에서 현실성 측정. AAAI, 2021.

[5] Kanjar De and V. Masilamani. 주파수 영역에서 블러 이미지의 이미지 선명도 측정. Procedia Engineering, 2013.

[6] Jia Deng, Wei Dong, Richard Socher, et al. ImageNet: 대규모 계층적 이미지 데이터베이스. CVPR, 2009.

[7] Prafulla Dhariwal and Alexander Nichol. 확산 모델은 이미지 합성에서 GAN을 능가한다. NeurIPS, 2021.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. 이미지 하나는 16×16 단어만큼의 가치가 있다: 대규모 이미지 인식을 위한 트랜스포머. ICLR, 2020.

[9] Patrick Esser, Robin Rombach, and Bjorn Ommer. 고해상도 이미지 합성을 위한 트랜스포머 길들이기. CVPR, 2021.

[10] Shuyang Gu, Dong Chen, Jianmin Bao, et al. 텍스트-이미지 합성을 위한 벡터 양자화 확산 모델. CVPR, 2022.

[11] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. ICCV, 2017.

[12] Amir Hertz, Ron Mokady, Jay Tenenbaum, et al. 교차 어텐션 제어를 이용한 프롬프트 간 이미지 편집. arXiv, 2022.

[13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, et al. 두 시간 척도 업데이트 규칙으로 학습된 GAN은 국소 내시 균형으로 수렴한다. NeurIPS, 2017.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 잡음 제거 확률 확산 모델. NeurIPS, 2020.

[15] Jonathan Ho, Chitwan Saharia, William Chan, et al. 고충실도 이미지 생성을 위한 계층적 확산 모델. Journal of Machine Learning Research, 2022.

[16] Jonathan Ho and Tim Salimans. 분류기 없는 확산 유도. NeurIPS 워크숍, 2021.

[17] Emiel Hoogeboom and Tim Salimans. 블러 확산 모델. arXiv, 2022.

[18] Yifan Jiang, Shiyu Chang, and Zhangyang Wang. TransGAN: 순수 트랜스포머 두 개로 강력한 GAN을 만들고 확장하기. NeurIPS, 2021.

[19] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, et al. 생성 모델 평가를 위한 개선된 정밀도 및 재현율 지표. NeurIPS, 2019.

[20] Sangyun Lee, Hyungjin Chung, Jaehyeon Kim, and Jong Chul Ye. 거친 단계에서 세밀한 단계로의 이미지 합성을 위한 확산 모델의 점진적 디블러링. NeurIPS 워크숍, 2022.

[21] Chenlin Meng, Ruiqi Gao, Diederik P. Kingma, et al. 유도된 확산 모델의 증류에 관하여. arXiv, 2022.

[22] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter Battaglia. 희소 표현을 이용한 이미지 생성. ICML, 2021.

[23] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, et al. GLIDE: 텍스트 유도 확산 모델을 이용한 사실적인 이미지 생성 및 편집을 향하여. arXiv, 2021.

[24] Alexander Quinn Nichol and Prafulla Dhariwal. 개선된 잡음 제거 확률 확산 모델. ICML, 2021.

[25] Bjorn Ommer and Joachim M. Buhmann. 시각 객체의 구성적 특성 학습. CVPR, 2007.

[26] Adam Paszke, Sam Gross, Francisco Massa, et al. PyTorch: 명령형 방식의 고성능 딥러닝 라이브러리. NeurIPS, 2019.

[27] William Peebles and Saining Xie. 트랜스포머를 이용한 확장 가능한 확산 모델. arXiv, 2022.

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. 자연어 감독을 통한 전이 가능한 시각 모델 학습. ICML, 2021.

[29] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, et al. CLIP 잠재 변수를 이용한 계층적 텍스트 조건부 이미지 생성. arXiv, 2022.

[30] Severi Rissanen, Markus Heinonen, and Arno Solin. 역열 확산을 이용한 생성 모델링. arXiv, 2022.

[31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al. 잠재 확산 모델을 이용한 고해상도 이미지 합성. CVPR, 2022.

[32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: 생의학 이미지 분할을 위한 합성곱 네트워크. MICCAI, 2015.

[33] Chitwan Saharia, William Chan, Saurabh Saxena, et al. 심층 언어 이해를 갖춘 사실적 텍스트-이미지 확산 모델. arXiv, 2022.

[34] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, et al. GAN 학습을 위한 개선된 기법. NeurIPS, 2016.

[35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 비평형 열역학을 이용한 비지도 학습. ICML, 2015.

[36] Jiaming Song, Chenlin Meng, and Stefano Ermon. 잡음 제거 확산 암시적 모델. ICLR, 2021.

[37] Yang Song and Stefano Ermon. 데이터 분포의 기울기 추정을 통한 생성 모델링. NeurIPS, 2019.

[38] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, et al. 확률 미분방정식을 통한 점수 기반 생성 모델링. ICLR, 2020.

[39] Zhicong Tang, Shuyang Gu, Jianmin Bao, et al. 개선된 벡터 양자화 확산 모델. arXiv, 2022.

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention Is All You Need. NeurIPS, 2017.

[41] Tengfei Wang, Ting Zhang, Bo Zhang, et al. 이미지-이미지 변환에는 사전 학습만 필요하다. arXiv, 2022.

[42] Yiwen Xu, Maurice Pagnucco, and Yang Song. DHG-GAN: 분리된 고주파 의미론을 이용한 다양한 이미지 아웃페인팅. ACCV, 2022.

[43] Mengping Yang, Zhe Wang, Ziqiu Chi, and Wenyi Feng. WaveGAN: 고충실도 소수 샷 이미지 생성을 위한 주파수 인식 GAN. ECCV, 2022.

[44] Fisher Yu, Ari Seff, Yinda Zhang, et al. LSUN: 인간 참여형 딥러닝을 이용한 대규모 이미지 데이터셋 구축. arXiv, 2015.

[45] Bowen Zhang, Shuyang Gu, Bo Zhang, et al. StyleSwin: 고해상도 이미지 생성을 위한 트랜스포머 기반 GAN. CVPR, 2022.

[46] Han Zhang, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. 자체 어텐션 생성적 적대 신경망. ICML, 2019.