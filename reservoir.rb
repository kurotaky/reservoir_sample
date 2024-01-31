require 'numo/narray'
require 'matrix'

class EchoStateNetwork
  def initialize(input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1, alpha=1e-4)
    @input_size = input_size
    @reservoir_size = reservoir_size
    @output_size = output_size
    @alpha = alpha  # Ridge regression regularization parameter

    # Initialize reservoir weights
    @reservoir_weights = Numo::DFloat.new(reservoir_size, reservoir_size).rand - 0.5
    mask = Numo::DFloat.new(reservoir_size, reservoir_size).rand > sparsity
    @reservoir_weights[mask] = 0

    # Scale weights with a fixed spectral radius
    @reservoir_weights *= spectral_radius / 1.25  # Adjust this scaling factor as needed

    @input_weights = Numo::DFloat.new(reservoir_size, input_size).rand - 0.5

    # Output weights (to be set after training)
    @output_weights = nil
  end

  def fit(x, y)
    states = Numo::DFloat.zeros(x.shape[0], @reservoir_size)
    x.shape[0].times do |t|
      states[t, true] = Numo::NMath.tanh(@input_weights.dot(x[t, true]) + @reservoir_weights.dot(states[t-1, true]))
    end

    # Numo::NArray から Ruby の Array に変換
    states_array = states.to_a.map { |row| row.to_a }
    y_array = y.to_a  # 1次元 Numo::NArray の場合

    states_matrix = Matrix[*states_array]
    reg_identity_matrix = Matrix.scalar(@reservoir_size, @alpha)

    # Train output weights using Ruby's Matrix for inverse calculation
    states_transpose = states_matrix.transpose
    @output_weights = (states_transpose * states_matrix + reg_identity_matrix).inverse * states_transpose * Matrix.column_vector(y_array)

    # 出力重みを Numo::NArray に戻す
    @output_weights = Numo::DFloat[*@output_weights.to_a.map(&:to_a)]
  end

  def predict(x)
    states = Numo::DFloat.zeros(x.shape[0], @reservoir_size)
    predictions = Numo::Int32.zeros(x.shape[0])
    x.shape[0].times do |t|
      states[t, true] = Numo::NMath.tanh(@input_weights.dot(x[t, true]) + @reservoir_weights.dot(states[t-1, true]))
      raw_output = states[t, true].dot(@output_weights)
      softmax_output = softmax(raw_output)
      predictions[t] = softmax_output.max_index
    end

    predictions
  end


  def softmax(x)
    e_x = Numo::NMath.exp(x - x.max)
    e_x / e_x.sum
  end

  def save(filename)
    File.open(filename, 'wb') { |file| Marshal.dump(@output_weights, file) }
  end
end


# データの生成
def generate_data(num_samples, num_features, num_classes)
  data = Numo::DFloat.zeros(num_samples, num_features)
  labels = Numo::Int32.zeros(num_samples)

  num_samples.times do |i|
    class_index = i % num_classes
    range_start = class_index.to_f / num_classes
    range_end = (class_index + 3).to_f / num_classes
    features = Numo::DFloat.new(num_features).rand * (range_end - range_start) + range_start
    data[i, true] = features
    labels[i] = class_index
  end

  return data, labels
end

# 混同行列の計算
def confusion_matrix(true_labels, predicted_labels, num_classes)
  matrix = Numo::Int32.zeros(num_classes, num_classes)
  true_labels.size.times do |i|
    matrix[true_labels[i], predicted_labels[i]] += 1
  end
  matrix
end

# パラメータ
num_classes = 3
num_features = 10  # 特徴量の数
num_samples_train = 300  # 学習データ数 (100 * 3)
num_samples_test = 30  # テストデータ数 (10 * 3)
reservoir_size = 500  # リザバーのサイズ

# データ生成
train_data, train_labels = generate_data(num_samples_train, num_features, num_classes)
test_data, test_labels = generate_data(num_samples_test, num_features, num_classes)

# ネットワークの初期化と学習
esn = EchoStateNetwork.new(num_features, reservoir_size, num_classes)
esn.fit(train_data, train_labels)

# テストデータで評価
 predicted_labels = esn.predict(test_data)

# 正解と予測の比較（要素単位）
correct_predictions = predicted_labels.eq(test_labels)

puts "Predicted labels range: #{predicted_labels.min} to #{predicted_labels.max}"
puts "Test labels range: #{test_labels.min} to #{test_labels.max}"

# 正解の数をカウント
num_correct = correct_predictions.count_true

# 正確さの計算
accuracy = num_correct.to_f / num_samples_test
puts "Accuracy: #{accuracy * 100}%"


# 混同行列の計算
matrix = confusion_matrix(test_labels, predicted_labels, num_classes)

# 混同行列の表示
puts "Confusion Matrix:"
puts matrix.to_a.map { |row| row.join(' ') }.join("\n")
