import numpy as np

from src.ga.generation import Generation

# 重さ、価値を乱数で100個生成する
# 容量は総重量の半分
WEIGHT = np.random.randint(10, 100, 100)
VALUE = np.random.randint(1, 50, 100)
CAPACITY = np.sum(WEIGHT) / 2

GEN_NUM = 50                # 世代数
INDIVIDUAL_NUM = 100        # 個体数
GENE_LENGTH = len(WEIGHT)   # 遺伝子長


# 制約条件の判定を行う関数
def eval_func(gene):
    return sum(WEIGHT * np.array(gene)) <= CAPACITY


# 適応度の計算を行う関数
def fitness_func(gene):
    return 0 if not eval_func(gene) else sum(VALUE * np.array(gene))


# メイン処理
def main():
    print("Weight: {}".format(WEIGHT))
    print("Value: {}".format(VALUE))
    print("Capacity: {}".format(CAPACITY))
    print("Generation: {}".format(GEN_NUM))
    print("Individuals: {}\n".format(INDIVIDUAL_NUM))

    gen = Generation(INDIVIDUAL_NUM, GENE_LENGTH, fitness_func)

    for i in range(GEN_NUM):
        gen.print_gen(i)        # 現世代の最大値とその個体を表示
        gen.nextgen()           # 次世代を生成

    return


if __name__ == '__main__':
    main()