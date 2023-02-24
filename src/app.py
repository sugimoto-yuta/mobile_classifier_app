import torch
from model import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルを元に推論する
def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('./feature_smart_classifier.pt', map_location=torch.device('cpu')))
    img = transform(img)
    # 1次元増やす
    img = img.unsqueeze(0)
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# 推論したラベルから犬か猫かを返す関数
def getName(label):
    if label == 0:
        return 'ガラケー'
    elif label == 1:
        return 'スマホ'
    
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URLにアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):
            # 画像ファイルに対する処理
            # 画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            # 画像データをバッファに読み込む
            image.save(buf, 'png')
            # バイナリーデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            # HTML側のの記述に合わせるために付帯情報を付与
            base64_data = f'data:image/png;base64,{base64_str}'

            # 入力された画像に対して推論
            pred = predict(image)
            mobileClass_ = getName(pred)
            return render_template('result.html', mobileClass=mobileClass_, image=base64_data)
        return redirect(request.url)

    elif request.method == 'GET':
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)