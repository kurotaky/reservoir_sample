require 'ostruct'
require 'numo/narray'
require 'numo/linalg'

class ESN
  attr_accessor :c, :is_feedback, :is_static_node, :Wr, :Wi, :Wb, :Wo, :X, :Y

  def initialize(config)
    @c = config
    Numo::NArray.srand(@c.seed) if @c.seed
    @n_classes = 3  # クラス数
    @Wo = Numo::DFloat.zeros(@c.Nx + 1, @n_classes)  # 出力重みの初期化
  end

  def one_hot_encode(y, n_classes)
    y_one_hot = Numo::DFloat.zeros(y.size, n_classes)
    y.each_with_index do |val, idx|
      y_one_hot[idx, val.to_i - 1] = 1.0  # クラスラベルが1から始まると仮定
    end
    y_one_hot
  end

  def fit(x, y, lambda_ridge)
    y_encoded = one_hot_encode(y, @n_classes)
    x_with_bias = Numo::DFloat.hstack([x, Numo::DFloat.ones([x.shape[0], 1])])
    @Wo = compute_ridge_regression(x_with_bias, y_encoded, lambda_ridge)
  end

  def compute_ridge_regression(x, y, lambda)
    xt = x.transpose
    xt_x = xt.dot(x)
    identity = Numo::DFloat.eye(x.shape[1])
    ridge_term = identity * lambda
    inverse = Numo::Linalg.inv(xt_x + ridge_term)
    inverse.dot(xt).dot(y)
  end

  def softmax(x)
    exp_x = Numo::NMath.exp(x - x.max(axis: 1).expand_dims(1))
    sum_exp_x = exp_x.sum(axis: 1).expand_dims(1)
    exp_x / sum_exp_x
  end

  def predict(x)
    x_with_bias = Numo::DFloat.hstack([x, Numo::DFloat.ones([x.shape[0], 1])])
    y_pred = x_with_bias.dot(@Wo)

    # probs の各要素の中にある3つの数値のうち、１番大きいもののインデックスを返す
    # つまり、各データに対して、最も確率の高いクラスを返す
    # そのため、最終的な予測結果は、各データに対して、最も確率の高いクラスのインデックスを返す
    # ただし、インデックスは0から始まるため、1を足してクラスラベルに変換する
    probs = softmax(y_pred)
    preds = []
    probs.shape[0].times.map do |i|
      preds << probs[i, true].max_index + 1
    end
    preds
  end
end

# ESN クラスのインスタンスを生成
config = OpenStruct.new(
  seed: 42,
  Nx: 3,  # 特徴量の数
  Nu: 1,
  Ny: 1,
  alpha_r: 1.0,
  beta_r: 0.0,
  alpha_i: 1.0,
  beta_i: 0.0,
  alpha_b: 1.0,
  beta_b: 0.0
)
esn = ESN.new(config)

# 特徴量とラベルの準備
train_data = [[1.2,1,3,1], [1.22,2,1,1], [1.2,2.13,1.2,1], [1.11,2.1,1,1], [1.22,1,1,1],
              [0.22,1.88,1,2], [0.12,1.98,0.9,2], [0.22,1.88,1,2], [0.21,1.78,1,2], [0.3,1.89,1,2],
              [0.22,0.88,2.1,3], [0.22,1.18,2.01,3], [1.02,1.08,2.1,3], [0.22,0.88,2.4,3], [0.22,0.68,2.1,3],
             ]
features = train_data.map { |row| row[0..2] }
labels = train_data.map { |row| row[3] }

x = Numo::DFloat[*features]
y = Numo::DFloat[labels].reshape(labels.size, 1)

# モデルのトレーニング
lambda_ridge = 0.1
esn.fit(x, y, lambda_ridge)

# 新しいデータに対する予測
test_data = [[1.22,1.38,3,1], [1.25,2,1,1], [1.1,2.13,1.2,1], [1.11,2.1,1,1], [1.22,1,1,1],
             [0.21,1.88,1,2], [0.11,1.98,0.9,2], [0.23,1.88,1,2], [0.21,1.78,1,2], [0.3,1.89,1,2],
             [0.22,0.88,2.1,3], [0.22,1.18,2.01,3], [1.02,1.08,2.1,3], [0.22,0.88,2.4,3], [0.22,0.68,2.1,3],
            ]
test_features = test_data.map { |row| row[0..2] }
test_x = Numo::DFloat[*test_features]
predictions = esn.predict(test_x)

puts "Predictions:"
p predictions

predicted_labels = predictions # この部分を実際の予測結果に置き換えてください。

# 混同行列の初期化
n_classes = 3
confusion_matrix = Array.new(n_classes) { Array.new(n_classes, 0) }

# 混同行列を計算
labels.each_with_index do |actual_class, i|
  predicted_class = predicted_labels[i]
  confusion_matrix[actual_class - 1][predicted_class - 1] += 1
end

# 混同行列を表示
puts "Confusion Matrix:"
confusion_matrix.each do |row|
  puts row.join(' ')
end

# 正答率を計算
correct_predictions = confusion_matrix.each_with_index.sum do |row, i|
  row[i]
end
total_predictions = labels.size
accuracy = (correct_predictions.to_f / total_predictions) * 100

# 正答率を表示
puts "Accuracy: #{accuracy.round(2)}%"
