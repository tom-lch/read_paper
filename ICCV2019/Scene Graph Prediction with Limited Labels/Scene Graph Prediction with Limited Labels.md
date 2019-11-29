Scene Graph Prediction with Limited Labels

文章作者：

Vincent S. Chen

Paroma Varma

Ranjay Krishna

Michael Bernstein

Christopher Re ́

Li Fei-Fei

`Stanford University`

摘要
`
Visual knowledge bases such as Visual Genome power numerous applications in computer vision, including visual question answering and captioning, but suffer from sparse, incomplete relationships. All scene graph models to date are limited to training on a small set of visual relationships that have thousands of training labels each. Hiring human annotators is expensive, and using textual knowledge base completion methods are incompatible with visual data. In this paper, we introduce a semi-supervised method that as- signs probabilistic relationship labels to a large number of unlabeled images using few labeled examples. We analyze visual relationships to suggest two types of image-agnostic features that are used to generate noisy heuristics, whose out- puts are aggregated using a factor graph-based generative model. With as few as 10 labeled examples per relation- ship, the generative model creates enough training data to train any existing state-of-the-art scene graph model. We demonstrate that our method outperforms all baseline ap- proaches on scene graph prediction by 5.16 recall@100 for PREDCLS. In our limited label setting, we define a complexity metric for relationships that serves as an indi- cator (R2 = 0.778) for conditions under which our method succeeds over transfer learning, the de-facto approach for training with limited labels.
`

先说一说我对本文的理解：

本篇论文给我最大的感觉是不够好，没错，是展示的效果不够好，一方面是识别范围有限，另一方面是使用的模型太复杂。我个人认为在机器学习领域，
越简单的模型所能达到预期目标就也好。
但一篇ICCV总有一些亮点，本文中使用空间和种类等相关参数来训练模型，
实现了能够识别<主题-do-客体>并在图片的的标注出来。
在文中，用到了Mask R-CNN（这一模型是一个基于Faster R-CNN改进的模型，当然，Faster R-CNN本身也足够优秀）来预训练未标注数据。
使用空间距离为参数，将场景图中一些主体、客体以空间划分的无标注的数据能够有效的判断出来，并且能顾取得不错的效果。

本文的思想是使用半监督学习，使用现有的模型来达到将原始图片找出主体、客体区域框，来进一步的实现标注。
    
    
文中设计的两大思想：
类别特征categorical features以及空间特征spatial features

文中场景图向分析的算法
Algorithm 1 Semi-supervised Alg. to Label Relationships
````
1: INPUT: {(o, p, o′) ∈ Dp}∀p ∈ P — A small dataset of object pairs (o, o′) with multi-class labels for predicates.
2: INPUT: {(o, o′ )} ∈ DU } — A large unlabeled dataset of images with ob- jects but no relationship labels.
3: INPUT:f(·,·)—A function that extracts features from apair of objects.
4: INPUT:DT(·)—A decision tree.
5: INPUT: G(·) — A generative model that assigns probabilistic labels given
multiple labels for each datapoint
6: INPUT:train(·)—Function used to train a scene graph detection model.
7: Extract features and labels,Xp,Yp :={f(o,o′),pfor(o,p,o′)∈Dp},
XU :={(f(o,o′)for(o,o′)∈DU}
8: Generate heuristics by fitting J decision trees DTf it (Xp )
9: Assign labels to (o,o′)∈DU,Λ=DTpredict(XU) for J decision trees.
10: Learn generative model G(Λ) and assign probabilistic labels YU :=G(Λ)
11: Train scene graph model, SGM := train(Dp + DU , Yp + YU )
12: OUTPUT:SGM(·)
````

![图片: img/SPG0.png](https://github.com/wowowoll/read_paper/tree/master/img/SGP0.png)

![图片: img/SPG1.png](https://github.com/wowowoll/read_paper/tree/master/img/SGP1.png)
难点
````
πφ(Λ,Y) =1/Zφ (exp (􏰀φTΛY)

Lθ =EY∼π􏰂[log􏰀1+exp(−θTVTY)􏰁􏰃]

````







