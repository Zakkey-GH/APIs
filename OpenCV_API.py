import cv2 #OpenCVライブラリをPythonで使用するためのモジュール
import numpy as np #数値計算を効率的に行うためのライブラリ。特に配列計算。尚、Pythonの標準ライブラリではない

class ObjectDetector:
    def __init__(self, weights_path, config_path, classes_path):
        # YOLOモデルの読み込み と初期化
        self.net = cv2.dnn.readNet(weights_path, config_path)
        # クラス名のリストを読み込み
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 出力層の名前を取得
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # 各クラスに対してランダムな色を生成
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, frame):
        height, width, _ = frame.shape
        # 画像の前処理：サイズ変更、スケーリング、ブロブ化
        # サイズ変更：画像の縦横のピクセル数を変えるプロセス。画像サイズを統一する
        # スケーリング：画像のピクセル値を特定の範囲に変換するプロセス。一般的には、ピクセル値を0から1の範囲に正規化
        # ブロブ化：画像データをモデルに入力できる形式に変換するためのプロセス。バッチ処理や異なる入力形式に対応しやすくなる
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        # オブジェクト検出の実行
        outs = self.net.forward(self.output_layers)

        # 検出結果の格納リスト
        class_ids = []
        confidences = []
        boxes = []
        
        # 検出されたオブジェクトの処理
        for out in outs: #outsから取り出された要素が順に変数outに格納される
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # 信頼度が0.5以上の検出のみ処理
                if confidence > 0.5:
                    # バウンディングボックスの座標計算
                    # バウンディングボックスは、オブジェクト検出において対象物を囲む矩形のこと
                    # 座標計算は、画像中の物体の位置を特定し、その矩形の座標を求めるために使用
                    # 通常、バウンディングボックスは4つの座標で表す
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression（重複した検出の除去）の適用
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # 検出結果を画像に描画
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                confidence = confidences[i]
                # バウンディングボックスの描画
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # ラベルとスコアの描画
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, [(self.classes[class_ids[i]], confidences[i]) for i in range(len(boxes)) if i in indexes]

def main():
    # YOLOモデルの設定ファイルパス
    WEIGHTS_PATH = 'yolov3.weights'
    CONFIG_PATH = 'yolov3.cfg'
    CLASSES_PATH = 'coco.names'
    
    # カメラの初期化
    # 他アプリケーションでカメラ使用を一旦停止すること！！
    cap = cv2.VideoCapture(0)
    detector = ObjectDetector(WEIGHTS_PATH, CONFIG_PATH, CLASSES_PATH)
    paused = False

    # メインループ：リアルタイム検出
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            # オブジェクト検出の実行と結果の表示
            frame_with_detection, detected_objects = detector.detect_objects(frame)
            cv2.imshow('Real-time Object Detection', frame_with_detection)

            # 検出されたオブジェクトの表示
            if detected_objects:
                print("検出されたオブジェクト:")
                for obj, conf in detected_objects:
                    print(f"- {obj} (信頼度: {conf:.2f})")

        # キー入力の処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 終了
            break
        elif key == ord('p'):  # 一時停止
            paused = True
        elif key == ord('r'):  # 再開
            paused = False

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

def download_yolo_files():
    # YOLOv3の必要なファイルをダウンロード
    print("YOLOv3ファイルのダウンロードを開始します...数分かかります")
    print("重みファイルをダウンロード中...数分かかります")
    import urllib.request
    urllib.request.urlretrieve(
        "https://pjreddie.com/media/files/yolov3.weights", 
        "yolov3.weights"
    )
    print("設定ファイルをダウンロード中...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", 
        "yolov3.cfg"
    )
    print("クラスファイルをダウンロード中...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", 
        "coco.names"
    )
    print("ダウンロード完了! 映像ウィンドウが出ない場合、他アプリケーションでのカメラ使用を一旦停止し、再実行！！")

if __name__ == "__main__":
    download_yolo_files()
    main()