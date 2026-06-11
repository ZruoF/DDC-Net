
#!/bin/bash
#baseline: baseline_scratch_e20
#
#full: full_focal_e20
# 配置参数
PROJECT=runs_ddc_v8
NAME=full_v8_e200_l_640
LOG_PATH="log/$PROJECT/$NAME/train.log"

# 自动创建日志目录
mkdir -p "log/$PROJECT/$NAME"

# 打印启动信息
echo "🚀 开始训练：$NAME"
echo "📁 日志路径：$LOG_PATH"
echo "📅 时间：$(date)"

# baseline 执行训练命令
#yolo detect train \
#  model=ultralytics/cfg/models/11/yolo11.yaml \
#  data=data/visdrone.yaml \
#  imgsz=640 batch=16 epochs=200 \
#  optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 cos_lr=True \
#  amp=True seed=0 pretrained=False \
#  project="$PROJECT" name="$NAME" \
#  2>&1 | tee "$LOG_PATH"


# full+focal 执行训练命令
yolo detect train \
  model=ultralytics/cfg/models/v8/yolo8l-ddc-FULL.yaml \
  data=data/visdrone.yaml \
  imgsz=640 batch=8 epochs=200 use_focal_ciou=True\
  optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 cos_lr=True \
  amp=true seed=0 pretrained=False \
  project="$PROJECT" name="$NAME" \
  2>&1 | tee "$LOG_PATH"

# 训练结束提示
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "✅ 训练完成！日志已保存至：$LOG_PATH"
else
  echo "❌ 训练失败，请检查日志：$LOG_PATH"
fi
