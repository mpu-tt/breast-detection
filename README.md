## 乳腺通用分割模型

## 推理



## 训练（可选）

1. 数据集基于[remekkinas/rsna-roi-detector-annotations-yolo](https://www.kaggle.com/datasets/remekkinas/rsna-roi-detector-annotations-yolo)
2. 模型基于[ultralytics/yolov5-v7.0](https://github.com/ultralytics/yolov5/tree/v7.0)
3. 预训练权重基于[yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)

```python
python train.py --weights yolov5s.pt \
                --cfg models/yolov5s-breast.yaml \
                --data dataset/data.yaml \
                --hyp data/hyps/hyp.breast.yaml \
                --epochs 100 \
                --batch-size 512 \
                --imgsz 640 \
                --device 0
```

## 其它资源

1. 乳腺检测标注数据集：https://www.kaggle.com/datasets/remekkinas/rsna-roi-detector-annotations-yolo
2. 分割好的乳腺数据集：https://www.kaggle.com/datasets/remekkinas/rsna-breast-cancer-detection-poi-images
