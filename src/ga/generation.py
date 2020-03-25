import numpy as np

from .individual import Individual

# 世代クラス
class Generation:
    def __init__(self, individual_num, gene_length, fitness_func):
        self.individuals = [None] * individual_num      # 個体の集合
        self.nexts = [None] * individual_num            # 次世代の集合
        self.individual_num = individual_num            # 個体数

        # 個体集合を初期化
        for i in range(self.individual_num):
            self.individuals[i] = Individual(gene_length, fitness_func)

        # 初期適応度を計算
        self.eval()

    # 適応度の最大値とそれを実現する個体を表示
    def print_gen(self, gen, f):
        fitnesses = list(map(lambda x: x.get_fitness(), self.individuals))
        genes = list(map(lambda x: x.get_gene(), self.individuals))

        max_idx = np.argmax(fitnesses)

        print("{}# Max Fitness: {}; Individual: {}".format(gen, fitnesses[max_idx], genes[max_idx]), file=f)

    # 全個体の適応度を評価
    def eval(self):
        for i in range(self.individual_num):
            self.individuals[i].calc_fitness()

    # 次世代を生成
    def nextgen(self):
        self.nexts = self.individuals

        for i in range(self.individual_num):
            idx1, idx2 = self.selection(), self.selection()
            child = self.cross(idx1, idx2)
            self.nexts[i].set_gene(child)

        self.individuals = self.nexts
        self.eval()

    # 個体の選択をルーレット選択で実施する
    def selection(self):
        prob = list(map(lambda x: x.get_fitness(), self.individuals))
        prob = np.array(prob) / sum(prob)

        cumprob = np.cumsum(prob)
        rand = np.random.rand()

        selected_idx = 0

        for i in range(len(cumprob)):
            if rand < cumprob[i]:
                selected_idx = i
                break

        return selected_idx

    # 交差を一様交差で行う
    # cross_probは交差確率, mutate_probは突然変異確率
    def cross(self, idx1, idx2, cross_prob=0.5, mutate_prob=0.01):
        gene1 = self.individuals[idx1].get_gene()
        gene2 = self.individuals[idx2].get_gene()

        size = len(gene1)

        cross_mask = np.random.rand(size) <= cross_prob

        child = gene1
        child[cross_mask] = gene2[cross_mask]

        child = self.mutate(child, mutate_prob=mutate_prob)

        return child

    # 突然変異を実行
    def mutate(self, child, mutate_prob=0.01):
        mutate_mask = np.random.rand(len(child)) <= mutate_prob

        child[mutate_mask] = np.abs(child[mutate_mask] - 1)

        return child