import cv2
import numpy as np
import json
import subprocess

def extract_color(image):
    """HSV色空間で平均色を抽出する関数"""
    resized = cv2.resize(image, (320, 240))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mean_color = np.mean(hsv, axis=(0, 1)).tolist()
    return mean_color

def extract_shape(image):
    """輪郭の頂点数を抽出する関数"""
    resized = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
        return len(approx)
    return 0

def extract_texture(image):
    """ラプラシアンフィルタでテクスチャを抽出する関数"""
    resized = cv2.resize(image, (320, 240))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(laplacian)))

def classify_trash(frame):
    """特徴を抽出してJuliaで分類を行う関数"""
    features = {
        "color": extract_color(frame),
        "shape": extract_shape(frame),
        "texture": extract_texture(frame)
    }
    
    # 特徴をJSONファイルに保存
    with open("features.json", "w") as f:
        json.dump(features, f)
    
    # Juliaスクリプトを呼び出し
    result = subprocess.run(["julia", "classify_trash.jl"], capture_output=True, text=True)
    return result.stdout.strip()

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 分類を実行
        result = classify_trash(frame)

        # 結果を表示
        cv2.putText(frame, f"Classification: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-time Trash Classification", frame)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()