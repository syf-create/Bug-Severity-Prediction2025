# -*- coding: utf-8 -*-
import numpy as np
import random


class DEAlgorithm:
    def __init__(self,config):
        self.config = config
        self.mutation_factors = []  # 保存每个试验向量的变异因子
        self.crossover_probabilities = []  # 保存每个试验向量的交叉概率
    def generate_F_Cr(self, μ_F, μ_Cr):
        '''
        Generate mutation factor F and crossover rate Cr
        :param μ_F: Mean for mutation factor F
        :param μ_Cr: Mean for crossover rate Cr
        :return: tuple (F, Cr)
        '''
        # Generate F and Cr from normal distribution with mean μ_F, μ_Cr and std 0.1
        F = np.random.normal(loc=μ_F, scale=0.1)
        Cr = np.random.normal(loc=μ_Cr, scale=0.1)

        # Clip F to be in the range [0, 2] and Cr to be in the range [0, 1]
        F = np.clip(F, 0, 2)
        Cr = np.clip(Cr, 0, 1)

        return F, Cr

    def initialization(self,initial_popusize):
        '''
        initialize population
        :return: initialized population
        '''
        ArryLi = []
        for i in range(initial_popusize):
            initLi = []
            for bound in self.config.bounds_all:
                init = np.random.uniform(low=bound[0], high=bound[1], size=1)
                initLi.append(init[0])
            ArryLi.append(initLi)

        return ArryLi

    def listminus(self,l1,l2):
        '''
        two list minus
        :param l1: list 1
        :param l2: list 2
        :return: minus list
        '''
        out = [a-b for a,b in zip(l1,l2)]
        return out

    def listadd(self,l1,l2):
        '''
        function to add two list
        :param l1: list1
        :param l2: list2
        :return: added list
        '''
        out = [a+b for a,b in zip(l1,l2)]
        return out

    def multply(self,a,li):
        '''
        function to multply two lists a*list
        :param a: a scalar
        :param li: a list
        :return: a list
        '''
        out = [a*b for b in li]
        return out

    def checkin(self,value,bound):
        if value>=bound[0] and value<=bound[1]:
            return True
        else:
            return False

    def inrange(self,list):
        '''
        determine the initialized population is in the bounds
        :param mutated: population after mutation
        :return: True or False
        '''
        logicli = []
        for i in range(len(list)):
            value = list[i]
            bound = self.config.bounds_all[i]
            logic = self.checkin(value,bound)
            logicli.append(logic)
        logic_ = all(logicli)

        return logic_

    def mutation(self, InitialArray, μ_F, μ_Cr):
        '''
        Mutation function
        :param InitialArray: initialized population
        :param μ_F: mean mutation factor
        :param μ_Cr: mean crossover rate
        :return: mutated population
        '''
        ArryLi = InitialArray.copy()
        for i in range(len(ArryLi)):
            Contin = True
            while Contin:
                # 使用 generate_F_Cr 获取 F 和 Cr
                F, Cr = self.generate_F_Cr(μ_F=μ_F, μ_Cr=μ_Cr)

                # 保存当前的突变因子和交叉概率
                self.mutation_factors.append(F)
                self.crossover_probabilities.append(Cr)

                # Perform mutation operation
                idxlist = list(range(len(ArryLi)))
                idxlist.remove(i)
                idxLI = random.sample(idxlist, 3)
                mu_a = self.listminus(ArryLi[idxLI[1]], ArryLi[idxLI[2]])
                mu_b = self.multply(F, mu_a)
                mu_c = self.listadd(ArryLi[idxLI[0]], mu_b)

                if self.inrange(mu_c):
                    ArryLi[i] = mu_c
                    Contin = False

        return ArryLi

    def crossover(self,InitialArray,MutatedArray,μ_F,μ_Cr):
        '''
        crossover function
        :param InitialArray: initialized population
        :param MutatedArray: mutated population
        :return: crossovered population
        '''
        CrossArray = []
        for i in range(len(InitialArray)):
            arry = []
            for j in range(len(InitialArray[i])):
                # 使用 generate_F_Cr 获取 Cr
                _, Cr = self.generate_F_Cr(μ_F=μ_F, μ_Cr=μ_Cr)  # 假设 μ_Cr 传入原始的 Cr 值

                if Cr >= self.config.F_c:
                    arry.append(MutatedArray[i][j])
                else:
                    arry.append(InitialArray[i][j])

            CrossArray.append(arry)

        return CrossArray




    def selection(self, gmeanlist, InitialArray, CrossArray, k):
        '''
        selection function
        :param gmeanlist: fitness list
        :param InitialArray: initialized population
        :param CrossArray: crossovered population
        :return: selectArray: selected population; Bestgmean: best fitness value (gmean)
        '''
        Bestgmean = []
        selectArray = []
        for i in range(k):
            gmeanIni = gmeanlist[i]
            gmeanCros = gmeanlist[i + k]
            if gmeanIni < gmeanCros:
                Bestgmean.append(gmeanCros)
                selectArray.append(CrossArray[i])
            else:
                Bestgmean.append(gmeanIni)
                selectArray.append(InitialArray[i])

        return selectArray, Bestgmean

    def get_mutation_factors(self):
        '''
        Return the list of mutation factors used for all trial vectors
        '''
        return self.mutation_factors

    def get_crossover_probabilities(self):
        '''
        Return the list of crossover probabilities used for all trial vectors
        '''
        return self.crossover_probabilities
