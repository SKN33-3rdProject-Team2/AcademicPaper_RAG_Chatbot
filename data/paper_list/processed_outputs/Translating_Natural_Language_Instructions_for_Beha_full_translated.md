# 원제: Translating Natural Language Instructions for Behavioral Robot Navigation with a Multi-Head Attention Mechanism

[본문 전체 마크다운 번역 결과]

# 다중 헤드 어텐션 메커니즘을 활용한 자연어 지침의 행동 기반 로봇 내비게이션 번역

Patricio Cerda-Mardini, Vladimir Araujo, Alvaro Soto  
Pontificia Universidad Catolica de Chile  
Millennium Institute for Foundational Research on Data  
{pcerdam, vgaraujo}@uc.cl, asoto@ing.puc.cl

## 초록

본 연구에서는 실내 로봇 내비게이션을 위해 자연어를 고수준 행동 언어로 번역하는 신경망 모델의 혼합 계층으로 다중 헤드 어텐션 메커니즘을 제안한다. 이를 위해 내비게이션 그래프를 과제의 지식 기반으로 활용하는 Zang et al. (2018a)의 프레임워크를 따른다. 실험 결과, 이전에 관찰되지 않은 환경에서 지침을 번역할 때 성능이 크게 향상되었으며, 이를 통해 모델의 일반화 능력이 개선됨을 확인했다.

## 배경

자연어 지침을 따를 수 있는 로봇 에이전트를 개발하는 것은 여전히 해결되지 않은 과제이다. 이상적으로 로봇은 사용자의 자연어 지침을 바탕으로 실행 가능한 내비게이션 계획을 정확하게 생성할 수 있어야 한다. 목표는 복잡하지만 사전에 알려진 실내 환경(그림 1(a))에서 출발지로부터 목적지에 도달하는 것이다. 이 환경은 그래프로 표현할 수 있으며(Sepulveda et al., 2018), 그래프의 노드는 사무실이나 침실과 같은 위치를, 간선은 복도를 따라가기나 사무실에서 나가기와 같이 인접한 노드 사이를 이동하게 하는 고수준 행동을 나타낸다(그림 1(b)). 본 연구에서는 로봇이 Sepulveda et al. (2018)에서와 같이 모든 고수준 행동을 안정적으로 실행할 수 있다고 가정한다.

기존 연구에서는 이 문제를 지침을 순차적으로 실행되는 고수준 행동 계획으로 번역하는 문제로 설정했으며(Zang et al., 2018b), 그래프 표현을 통해 환경의 위상 정보를 활용했다(Zang et al., 2018a). 구체적으로 지도학습 모델은 사용자의 텍스트 지침, 로봇의 초기 위치, 그리고 $(n_{1}, b, n_{2})$ 형태의 트리플릿으로 인코딩된 환경의 행동 그래프를 입력으로 받는다. 여기서 $n_{1}$과 $n_{2}$는 장소이고, $b$는 두 장소를 연결하는 행동이다. 모델은 일반적인 단일 소프트 어텐션 계층을 갖춘 시퀀스-투-시퀀스 모델을 통해 지시된 목적지에 도달하기 위한 행동 시퀀스를 예측한다. 이 어텐션 계층은 그래프 정보와 지침 정보를 결합한다.

그러나 추론 단계에서 이 접근법은 학습 중 관찰되지 않은 환경에 적용될 경우 심각한 성능 저하를 보인다. 본 연구에서는 어텐션 계층을 다중 헤드 메커니즘으로 수정하여 모델의 일반화 능력을 향상하고, 이에 따라 새로운 환경에서의 성능을 높이고자 한다.

> **그림 1: (a) 환경 지도. (b) 행동 내비게이션 그래프. (c) 제안 모델.** (c)의 자연어 지침은 순차적 행동 계획으로 번역된다. (a)에서 빨간색으로 강조된 경로와 (b)에서 빨간색으로 표시된 노드-간선은 모델 (c)가 예측한 행동에 해당한다.
>
> 자연어 지침: “방에서 나와 오른쪽으로 돌아서 꽃병을 지날 때까지 복도를 따라간 다음, 왼쪽에 있는 다음 방으로 들어가세요.”
>
> 모델은 GRU, 양방향 GRU(Bi-GRU), 다중 헤드 어텐션, 그리고 행동 시퀀스를 생성하는 GRU 디코더로 구성된다. 입력은 자연어 지침과 행동 그래프 트리플릿이며, 출력은 시작 위치에서 목표 위치까지의 행동 시퀀스이다.

## 방법론

## 접근법

Transformer 모델이 다중 모달 데이터 사이의 다양한 관계를 인코딩하는 데 성공한 점에 착안하여(Vaswani et al., 2017; Tan and Bansal, 2019; Zhou et al., 2020), 자연어 지침과 내비게이션 그래프라는 두 표현 부분공간의 정보를 보다 유용한 방식으로 결합하기 위해 Transformer의 다중 헤드 어텐션 메커니즘을 활용한다. 즉, 각 헤드는 두 정보원 사이의 서로 다른 패턴을 결합하는 데 특화된다. 이러한 능력이 디코더가 테스트 시 새로운 환경에서 겪는 성능 저하를 완화하는 데 도움이 될 것이라고 가정한다.

### 제안 모델

제안한 아키텍처(그림 1(c))는 초기 인코딩 계층으로 구성된다. 여기서 지침의 각 단어는 사전 학습된 GloVe 표현(Pennington et al., 2014)을 사용해 인코딩되며, 각 트리플릿 집합은 해당 트리플릿을 구성하는 $B$개의 행동과 $N$개의 노드 중 어떤 요소인지를 나타내도록 원-핫 인코딩된다. 이후 양방향 게이트 순환 유닛(GRU)(Chung et al., 2014)을 사용해 이러한 인코딩을 임베딩한다.

그다음 새롭게 추가된 다중 헤드 어텐션 메커니즘을 통해 두 모달리티의 표현을 결합한다. 결합된 정보는 후단의 완전연결 계층을 거치며 차원이 축소되고, 이렇게 얻은 표현은 순환 GRU 디코더의 문맥 $C$로 사용된다. 디코더는 초기 위치를 입력으로 받아 지침을 순차적 행동 계획으로 번역하며, 각 시점에서 문맥 $C$에 소프트 어텐션을 적용한다. 손실 함수는 정답 번역에 대한 교차 엔트로피이다.

## 실험 설정

Zang et al. (2018a)이 소개한 데이터셋과 기존의 학습 및 테스트 분할을 사용한다. Test-Repeated 분할에는 학습 단계에서 에이전트가 이미 관찰한 환경이 포함되며, Test-New 분할에는 이전에 관찰되지 않은 지도가 포함된다. 전체적으로 100개 지도에 걸쳐 10,040개의 지침을 사용하며, 이 중 8,066개는 학습에 사용된다. 각 지도에는 6개에서 65개의 방이 포함되어 있다.

성능 평가는 기존과 동일한 지표를 사용한다. 여기에는 F1 점수, 정답과의 편집 거리(ED), 그리고 M@k 지표가 포함된다. M@k는 번역 결과가 정답으로부터 $k$번의 이동 이내에 있을 때 일치하는 것으로 간주하는 지표이며, M@0은 정확히 일치하는 경우를 의미한다.^1

모델은 배치 크기 256으로 200 에포크 동안 학습했다. 다중 헤드 어텐션 계층의 헤드 수는 4개로 설정했다. 그 밖의 모델 매개변수는 Zang et al. (2018a)의 설정을 그대로 따랐다.

## 결과 및 논의

## 결과

표 1은 제안 방법의 성능을 Zang et al. (2018a)이 보고한 기준선 및 자체 구현한 기준선과 비교해 보여준다. 특히 자체 구현한 모델은 Test-Repeated 집합에서 예상한 수준의 성능을 내지 못했다.

다중 헤드 접근법을 사용한 결과, Test-New 집합의 정확 일치 성능이 23.2% 향상되었다. 이는 제안한 번역 모델의 일반화 능력이 개선되었음을 확인해 준다. 반면 Test-Repeated 집합에서는 기존 방식에 비해 정확 일치 성능이 8.5% 감소했다. 다만 이 집합에서는 자체 구현한 기준선보다 25.9%, Test-New 집합에서는 18.4% 더 높은 성능을 달성했다.

| 아키텍처 | Test-Repeated F1 ↑ | M@0 ↑ | M@1 ↑ | M@2 ↑ | ED ↓ | Test-New F1 ↑ | M@0 ↑ | M@1 ↑ | M@2 ↑ | ED ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 기준선(Zang) | **93.54** | **61.17** | **83.30** | **92.19** | **0.75** | 90.22 | 41.71 | 69.82 | 82.08 | 1.22 |
| 기준선(자체 구현) | 91.67 | 44.43 | 76.93 | 89.16 | 1.01 | 90.89 | 43.41 | 72.64 | 87.25 | 1.09 |
| 제안 방법 | 93.07 | 55.96 | 81.31 | 90.16 | 0.84 | **92.57** | **51.40** | **79.06** | **89.43** | **0.91** |

> **표 1: 결과.** ↑는 해당 열에서 값이 높을수록 우수함을, ↓는 값이 낮을수록 우수함을 의미한다.

## 결론

본 논문에서는 지식 기반을 활용하여 자연어를 로봇이 이해하고 실행할 수 있는 고수준 행동 언어로 번역하는 데 유용한 메커니즘으로 다중 헤드 어텐션을 도입했다. 제안 방법은 이전 연구와 비교하여 한 번도 관찰되지 않은 환경에서 더 우수한 성능을 보였다. 향후 연구에서는 이미 관찰한 지도에서의 성능 저하를 최소화하고, 생성된 어텐션 가중치에 대한 정성적 분석을 수행할 예정이다.

## 사사

본 연구는 Millennium Institute for Foundational Research on Data와 Fondecyt grant 1181739의 부분적인 지원을 받아 수행되었다.

^1 이동 한 번은 계획 내 행동 하나를 추가, 삭제 또는 교체하는 것을 의미한다.

## 참고문헌

Junyoung Chung, aglar G¨ulehre, Kyunghyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. *ArXiv*, abs/1412.3555.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In *EMNLP*.

Gabriel Sepulveda, Juan Carlos Niebles, and Alvaro Soto. 2018. A deep learning based behavioral approach to indoor autonomous navigation. In *2018 IEEE International Conference on Robotics and Automation (ICRA)*, pages 4646–4653. IEEE.

Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 5103–5114.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *31st Conference on Neural Information Processing Systems*, NIPS ’17.

Xiaoxue Zang, Ashwini Pokle, Marynel V´azquez, Kevin Chen, Juan Carlos Niebles, Alvaro Soto, and Silvio Savarese. 2018a. Translating navigation instructions in natural language to a high-level plan for behavioral robot navigation. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2657–2666, Brussels, Belgium, October–November. Association for Computational Linguistics.

Xiaoxue Zang, Marynel V´azquez, Juan Carlos Niebles, Alvaro Soto, and Silvio Savarese. 2018b. Behavioral indoor navigation with natural language directions. *Companion of the 2018 ACM/IEEE International Conference on Human-Robot Interaction*.

Yichao Zhou, Shaunak Mishra, Manisha Verma, Narayan Bhamidipati, and Wei Wang. 2020. Recommending themes for ad creative design via visual-linguistic representations.