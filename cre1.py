import numpy as np
from scipy import spatial
import tkinter as tk
from tkinter import messagebox, simpledialog
import itertools
import os
import random
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 初始化词向量字典
embeddings_dict = {}

# 获取资源文件路径（打包后路径）
def resource_path(relative_path):
    """获取资源文件的绝对路径（支持打包后运行）"""
    try:
        # PyInstaller打包时的临时路径
        base_path = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        # 没有打包时的正常路径
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 加载GloVe预训练词向量
def load_glove_embeddings(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word] = vector

# 计算两个词之间的余弦相似度
def cosine_similarity(word1, word2):
    if word1 not in embeddings_dict or word2 not in embeddings_dict:
        return None
    vector1 = embeddings_dict[word1]
    vector2 = embeddings_dict[word2]
    return 1 - spatial.distance.cosine(vector1, vector2)

# 找到与给定词相关系数最高的词
def most_similar_word(word):
    if word not in embeddings_dict:
        return None, None
    word_vector = embeddings_dict[word]
    max_similarity = 0
    most_similar = None
    for w, vector in embeddings_dict.items():
        if w == word:
            continue
        similarity = 1 - spatial.distance.cosine(word_vector, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = w
    return most_similar, max_similarity

# 计算用户输入的10个词的平均相关系数
def calculate_creativity(words):
    if len(words) != 10:
        return None
    similarities = []
    for word1, word2 in itertools.combinations(words, 2):
        similarity = 1-cosine_similarity(word1, word2)
        if similarity is not None:
            similarities.append(similarity)
    return np.mean(similarities) if similarities else None

# 生成随机词组的创造力分布
def generate_random_dat_scores(num_samples=100, num_words_per_sample=10):
    all_words = list(embeddings_dict.keys())
    random_dat_scores = []
    for _ in range(num_samples):
        random_words = random.sample(all_words, num_words_per_sample)
        score = calculate_creativity(random_words)
        if score is not None:
            random_dat_scores.append(score)
    return random_dat_scores

# 计算用户创造力在随机分布中的位置
def calculate_position(dat_score, random_dat_scores):
    random_dat_scores.sort()
    position = np.searchsorted(random_dat_scores, dat_score, side='right') / len(random_dat_scores)
    return position

# 可视化创造力分布
def visualize_creativity(dat_score, random_dat_scores):
    plt.hist(random_dat_scores, bins=20, alpha=0.7, color='blue', label='随机词的DAT分布')
    plt.axvline(dat_score, color='red', linestyle='dashed', linewidth=2, label=f'你的DAT ({dat_score*100:.4f})')
    plt.xlabel('DAT值') 
    plt.ylabel('频率')
    plt.title('创造力指数DAT分布')
    plt.legend()
    plt.show()

def visualize_words_with_tsne(words):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 提取单词的词向量
    word_vectors = []
    valid_words = []
    for word in words:
        if word in embeddings_dict:
            word_vectors.append(embeddings_dict[word])
            valid_words.append(word)
        else:
            print(f"警告: 单词 '{word}' 不在词向量字典中，将被跳过。")

    if not word_vectors:
        print("错误: 没有有效的单词可用于 t-SNE 可视化。")
        return

    # 将词向量列表转换为 NumPy 数组
    word_vectors = np.array(word_vectors)  # 关键修复点

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    Y = tsne.fit_transform(word_vectors)

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    plt.scatter(Y[:, 0], Y[:, 1], color='blue', label='Words')
    for label, x, y in zip(valid_words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.title('t-SNE Visualization of Input Words')
    plt.legend()
    plt.grid(True)
    plt.show()


# GUI界面
class WordVectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Vector Analysis")
        self.root.geometry("500x350")  # 调整窗口大小
        self.root.configure(bg="#f0f0f0")

        # 加载词向量
        self.load_embeddings()

        # 输入框和标签
        tk.Label(self.root, text="输入第一个词:", bg="#f0f0f0", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10)
        self.word1_entry = tk.Entry(self.root, font=("Arial", 12))
        self.word1_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.root, text="输入第二个词:", bg="#f0f0f0", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10)
        self.word2_entry = tk.Entry(self.root, font=("Arial", 12))
        self.word2_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(self.root, text="输入10个尽可能\n不相关的词（用逗号分隔）:", bg="#f0f0f0", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10)
        self.creativity_entry = tk.Entry(self.root, font=("Arial", 12))
        self.creativity_entry.grid(row=2, column=1, padx=10, pady=10)

        # 提示标签
        self.creativity_label = tk.Label(self.root, text="当前输入词数: 0/10", fg="red", bg="#f0f0f0", font=("Arial", 12))
        self.creativity_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # 按钮
        tk.Button(self.root, text="计算两个词的相似度", command=self.calculate_similarity, bg="#4CAF50", fg="white", font=("Arial", 12)).grid(row=4, column=0, padx=10, pady=10)
        tk.Button(self.root, text="找到与第一个词\n最相似的词", command=self.find_most_similar, bg="#008CBA", fg="white", font=("Arial", 12)).grid(row=4, column=1, padx=10, pady=10)
        self.assess_button = tk.Button(self.root, text="评估创造力", command=self.assess_creativity, bg="#f44336", fg="white", font=("Arial", 12), state=tk.DISABLED)
        self.assess_button.grid(row=5, column=0, padx=10, pady=10)
        tk.Button(self.root, text="开始输入", command=self.start_input, bg="#123456", fg="white", font=("Arial", 12)).grid(row=5, column=1, padx=10, pady=10)

        # 输出区域
        self.result_label = tk.Label(self.root, text="", fg="blue", bg="#f0f0f0", font=("Arial", 12))
        self.result_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        # 绑定输入框事件
        self.creativity_entry.bind("<KeyRelease>", self.update_creativity_label)

    def update_creativity_label(self, event=None):
        # 更新输入词数提示
        words = self.creativity_entry.get().strip().split(",")
        words = [word.strip() for word in words if word.strip()]
        word_count = len(words)
        self.creativity_label.config(text=f"当前输入词数: {word_count}/10")

        # 启用或禁用评估创造力按钮
        if word_count == 10:
            self.assess_button.config(state=tk.NORMAL)
        else:
            self.assess_button.config(state=tk.DISABLED)

    def load_embeddings(self):
        glove_file_path = resource_path("glove.6B.100d.txt")
        load_glove_embeddings(glove_file_path)
        messagebox.showinfo("加载完成", "程序可计算两个词之间的相似度，并通过输入10个尽可能不相关的词评估用户的创造力")

    def calculate_similarity(self):
        word1 = self.word1_entry.get().strip()
        word2 = self.word2_entry.get().strip()
        similarity = cosine_similarity(word1, word2)
        if similarity is None:
            messagebox.showerror("错误", "输入的词不在词表中！")
        else:
            self.result_label.config(text=f"{word1} 与 {word2} 的相似度为: {similarity:.4f}")

    def find_most_similar(self):
        word = self.word1_entry.get().strip()
        most_similar, similarity = most_similar_word(word)
        if most_similar is None:
            messagebox.showerror("错误", "输入的词不在词表中！")
        else:
            self.result_label.config(text=f"与 {word} 最相似的词是: {most_similar}，相似度为: {similarity:.4f}")

    def assess_creativity(self):
        words = self.creativity_entry.get().strip().split(",")
        words = [word.strip() for word in words]
        if len(words) != 10:
            messagebox.showerror("错误", "请输入10个词！")
        else:
            creativity_score = calculate_creativity(words)
            if creativity_score is None:
                messagebox.showerror("错误", "输入的词不在词表中！")
            else:
                self.result_label.config(text=f"10个词的相似度:{creativity_score:.4f},创造力评分: {100-creativity_score*100:.4f}")
                
                # 生成随机词组的创造力分布
                random_dat_scores = generate_random_dat_scores()
                position = calculate_position(creativity_score, random_dat_scores)
                messagebox.showinfo("创造力排名", f"你的创造力超越机器随机生成的：{(1-position)*100:.2f}%词组")
                
                # 可视化
                visualize_creativity(creativity_score, random_dat_scores)
                visualize_words_with_tsne(words)

    def start_input(self):
        self.word1_entry.delete(0, tk.END)
        self.word2_entry.delete(0, tk.END)
        self.creativity_entry.delete(0, tk.END)
        self.creativity_label.config(text="当前输入词数: 0/10")
        self.assess_button.config(state=tk.DISABLED)
        messagebox.showinfo("提示", "请开始输入！")

if __name__ == "__main__":
    root = tk.Tk()
    app = WordVectorGUI(root)
    root.mainloop()