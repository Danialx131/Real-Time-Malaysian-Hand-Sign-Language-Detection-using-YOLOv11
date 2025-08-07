from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def evaluate():
    # === Load your model ===
    model_path = r"D:\project\Real Time Malaysian Sign Language Detection Using YoloV11\Real Time Malaysian Sign Language Detection Using YoloV11 - project\yoloV8.pt"
    model = YOLO(model_path)

    # === Dataset path ===
    data_yaml = r"C:\Users\danny\PycharmProjects\yolov10\Dataset YoloV9\data.yaml"

    # === Evaluate ===
    metrics = model.val(data=data_yaml)

    print("\n📊 Evaluation Results:")
    print(f"mAP@0.5     : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"Precision   : {metrics.box.mp:.3f}")
    print(f"Recall      : {metrics.box.mr:.3f}")

    # 1. Confusion Matrix (saved as .png by Ultralytics)
    conf_matrix_path = os.path.join(metrics.save_dir, "confusion_matrix.png")
    if os.path.exists(conf_matrix_path):
        img = cv2.imread(conf_matrix_path)
        plt.figure(figsize=(7,7))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("⚠️ Confusion matrix not found.")

    # 2. Per-class mAP, Precision, Recall, F1 Score Bar Charts
    # -- Get class names --
    class_names = model.names if hasattr(model, "names") else [str(i) for i in range(metrics.box.nc)]
    indices = np.arange(len(class_names))
    width = 0.25

    # -- Per-class metrics --
    per_class_map50 = metrics.box.ap50  # AP@0.5 per class (array)
    per_class_map = metrics.box.ap  # AP@0.5:0.95 per class (array)
    per_class_prec = metrics.box.p  # precision per class (array)
    per_class_rec = metrics.box.r  # recall per class (array)

    per_class_f1    = 2 * (per_class_prec * per_class_rec) / (per_class_prec + per_class_rec + 1e-8)

    # mAP50 bar chart
    plt.figure(figsize=(9,4))
    plt.bar(indices, per_class_map50, width, label='mAP@0.5')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylim(0,1)
    plt.ylabel('mAP@0.5')
    plt.title('Per-class mAP@0.5')
    plt.tight_layout()
    plt.show()

    # Precision, Recall, F1 bar chart
    plt.figure(figsize=(9,4))
    plt.bar(indices-width, per_class_prec, width, label='Precision')
    plt.bar(indices, per_class_rec, width, label='Recall')
    plt.bar(indices+width, per_class_f1, width, label='F1 Score')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylim(0,1)
    plt.ylabel('Score')
    plt.title('Per-class Precision, Recall, F1')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. PR Curves (for each class, if available)
    if hasattr(metrics.box, "pr_curves"):
        pr_curves = metrics.box.pr_curves  # dict, class index -> (precision, recall)
        for i, cname in enumerate(class_names):
            if i in pr_curves:
                p, r = pr_curves[i]
                plt.plot(r, p, label=f'{cname}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Per-Class Precision-Recall Curves")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ PR curves not available in this version.")

if __name__ == '__main__':
    evaluate()
