import numpy as np


# 個体クラス
class Individual:
    def __init__(self, gene_length, fitness_func):
        self.gene = np.random.randint(0, 2, gene_length)        # 遺伝子
        self.fitness = 0                                        # 適応度
        self.fitness_func = fitness_func                        # 適応度を計算する関数

    # 遺伝子を更新する
    def set_gene(self, gene):
        if len(gene) != len(self.gene):
            raise Exception("length of gene should be equal to self.gene")

        self.gene = np.array(gene)

    # 適応度を計算する
    def calc_fitness(self):
        self.fitness = self.fitness_func(self.gene)

    # 遺伝子を取得する
    def get_gene(self):
        return self.gene

    # 適応度を取得する
    def get_fitness(self):
        return self.fitness
