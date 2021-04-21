#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:01:49 2021

@author: donutman
@licence : see LICENCE file

This module provides the main functions used for Littre's analysis (a French dictionary)


Usage
-----

As a main call, it will load the Littre object into 'littre' variable

As a module, it will export the Littre() class which contains the functions
used for Littre's analysis


Assumption
----------

The dictionary is stored in sparse matrix COO format in the following file
./data/littre_matrix.npz

The dictionary index is stored in text format (utf8) in the following file
./data/littre_entries.dat

The redundancy of each entries (aka the narcissistic number) is stored
in binary in the following file
./data/littre_narc.dat


Acknowledgment
--------------

The author wants to thank François Gannaz for having shared his XML Littre
under CC-by-SA 3.0 licence

Link : https://bitbucket.org/Mytskine/xmlittre-data/src/master/

Littré, Émile. Dictionnaire de la langue française. Paris, L. Hachette, 1873-1874.
Electronic version created by François Gannaz. http://www.littre.org

"""

from scipy import sparse
import numpy as np
from matplotlib import rc, pyplot as plt, patches as patches
import pandas as pd
import struct
import string


_file_matrix  = './data/littre_matrix.npz'
_file_entries = './data/littre_entries.dat'
_file_narc    = './data/littre_narc.dat'

# Global fontsize for plots
_fontsize = 22

rc('xtick', labelsize=_fontsize) 
rc('ytick', labelsize=_fontsize) 

class Littre():
    def __init__(self):
        self._littre = sparse.load_npz(_file_matrix)
        
        with open(_file_entries, 'rt') as f:
            self._names = [name.rstrip('\n ') for name in f.readlines()]
            
        with open(_file_narc, 'rb') as f:
            dat = f.read()
        self._narc = [narc[0] for narc in struct.iter_unpack('>H', dat)]
        self._narcvec = pd.Series(self._narc, index=self._names)
        self._narcvec.sort_values(inplace=True)
        self._narcvec.name = 'narcissistic counter'
        self._narcvec.index.name = '<word>'
        
        self._nentries = len(self._names)
        self._table = {n:k for (k, n) in enumerate(self._names)}

        # Definition length stuff
        self._deflen = np.squeeze(np.asarray(self._littre.sum(axis=1)))

        self._defvec = pd.Series(self._deflen, index=self._names)
        self._defvec.sort_values(inplace=True)
        self._defvec.name = 'definition length'
        self._defvec.index.name = '<word>'
        
        # Quotation number stuff (popularity)
        self._quote = np.squeeze(np.asarray(self._littre.sum(axis=0)))
        
        self._quotevec = pd.Series(self._quote, index=self._names)
        self._quotevec.sort_values(inplace=True)
        self._quotevec.name = 'popularity'
        self._quotevec.index.name = '<word>'
        
        self._letters = self._get_letters_rank()
        
    # Internal methods of Littre object
    def _get_outvec(self, rank_i):
        
        row = self._littre.getrow(rank_i)
        
        # rank_j contains all the indices for which row[rank_j] == True
        (_, rank_j, _) = sparse.find(row)

        #rank_j = [kc for (kr,kc) in zip(self._littre.row, self._littre.col) if kr==rank]
        return rank_j
    
    def _get_invec(self, rank_j):
        
        row = self._littre.getcol(rank_j)
        
        # rank_j contains all the indices for which row[rank_j] == True
        (rank_i, _, _) = sparse.find(row)

        #rank_j = [kc for (kr,kc) in zip(self._littre.row, self._littre.col) if kr==rank]
        return rank_i
    
    def _find_rank(self, letter):
        names_letter = [n for n in self._names if n.startswith(letter)]
        return self._table[names_letter[0]]
    
    def _get_letters_rank(self):
        letters = list(string.ascii_lowercase)
        ranks = []
        
        for i in letters:
            ranks.append(self._find_rank(i))
        
        return dict(zip(letters, ranks))
        
        
    
    
    # Textual methods of Littre object
    def get_definition(self, name):
        '''Returns the definition list of words for a given entry'''
        rank = self._table.get(name, -1)
        
        if rank >= 0: # the word has been found
            row = self._littre.getrow(rank)
            definitions = {self._names[k] for k in row.indices}
            return definitions
        else:
            print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
            return {}
    
    def print_most_narcissistic(self, number=10):
        '''Prints the most narcissistic words (aka entries that are self-quoted)'''
        print(self._narcvec.tail(number)[::-1])
        
    def print_least_narcissistic(self, number=10):
        '''Prints the least narcissistic words (aka entries that are self-quoted)'''
        print(self._narcvec.head(number))         
    
        
    def print_longest_def(self, number=10):
        '''Prints the entries with the maximum definition length'''
        print(self._defvec.tail(number)[::-1])
        
    def print_shortest_def(self, number=10):
         '''Prints the entries with the minimum definition length'''
        print(self._defvec.head(number))
        
    def print_most_popular(self, number=10):
         '''Prints the entries which are the most used in the definition
         of other entries'''
        print(self._quotevec.tail(number)[::-1])
        
    def print_least_popular(self, number=10):
         '''Prints the entries which are the least used in the definition
         of other entries'''
        print(self._quotevec.head(number))
        
    def get_outbeam(self, name):
        '''Returns the list out outbeam set of given entry. Basically, those
        are the indices of words in get_definition() function'''
        rank = self._table.get(name, -1)
        
        if rank >= 0: # the word has been found
            rank_j = np.array(self._get_outvec(rank))
            outbeam = []
            for k in rank_j:
                outbeam.append(self._names[int(k)])
            return outbeam
        else:
            print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
            return []    
        
    def get_inbeam(self, name):
        '''Returns the list out inbeam set of given entry. Basically, those
        are the indices of words that use the entry in their definition'''
        rank = self._table.get(name, -1)
        
        if rank >= 0: # the word has been found
            rank_i = np.array(self._get_invec(rank))
            inbeam = []
            for k in rank_i:
                inbeam.append(self._names[int(k)])
            return inbeam
        else:
            print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
            return []          
        
        
    # Graphics methods of Littre object
    def plot_def_len(self):
        '''Plots the definition length distribution'''
        
        letters = list('acehmrz')
        
        s_letters = []
        x_letters = []
        for l in letters:
            s_letters.append(l.upper())
            x_letters.append(self._letters[l])
        
        vmean = self._deflen.mean().round()
        vmax  = self._deflen.max()
        decim = 10 # Decimation factor
        
        hfig = plt.figure(figsize=[15,10])

        plt.subplot(2,1,1)
        plt.plot(np.arange(0, self._nentries, decim), self._deflen[::decim], 'k+')
        plt.ylabel('Definition length', fontsize=_fontsize)
        plt.xticks(ticks=x_letters, labels=s_letters)
        
        ax = plt.subplot(2,1,2)
        legend = 'Mean : %d words - Max : %d words' % (vmean, vmax)
        plt.hist(self._deflen, bins=range(200),color='r', label = legend)
        plt.xlabel('Definition length', fontsize=_fontsize)
        ax.legend(fontsize=_fontsize)
        plt.draw()
        
        return hfig
        
    def plot_adjacency_matrix(self):
        '''Plots the adjacency matrix of the dictionary'''
        n = self._nentries
        background = patches.Rectangle((0,0), n, n, facecolor='k', edgecolor='none')
        ms = 0.1
        decim = 5
        
        hfig, ax = plt.subplots(figsize=(15,15))
        ax.add_patch(background)
        
        #plt.plot([0,n], [0, n/2], 'r-+')
        plt.plot(self._littre.col[::decim], self._littre.row[::decim], 'w.', ms=ms)
        ax.invert_yaxis()
        ax.axis('equal')
        plt.axis('off')
        #ax.xaxis.set_visible(False)
        plt.title('Adjacency matrix', fontsize=_fontsize)
        
        return hfig
        
    def plot_outbeam(self, name):
        '''Graphic representation of outbeam (see get_outbeam() function above)'''
        
        letters = list('acehmrz')
        
        theta_letters = []
        for l in letters:
            theta_letters.append(self._letters[l]/(self._nentries+1)*2*np.pi)
        
        x_letters = np.cos(theta_letters)
        y_letters = np.sin(theta_letters)
            
        
        rank = self._table.get(name, -1)
        
        if rank >= 0: # the word has been found
            rank_j = np.array(self._get_outvec(rank))
        else:
            print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
            return {}
        
        theta0 = rank/(self._nentries+1)*2*np.pi
        x0 = np.cos(theta0)
        y0 = np.sin(theta0)
        
        theta = rank_j/(self._nentries+1)*2*np.pi
        x = np.cos(theta)
        y = np.sin(theta)
        
        C = patches.Circle((0,0), 1, edgecolor='k', facecolor=None, fill=False, lw=2)
        
        hfig, ax = plt.subplots()
        for (xk, yk) in zip(x, y):
            plt.plot([x0, xk], [y0, yk], 'k-')
        
        for (xl, yl, letter) in zip (x_letters, y_letters, letters):
            plt.plot(xl, yl, ls='', marker='o', mec = 'k', mfc = 'k', ms = 7)
            plt.text(1.15*xl, 1.15*yl, letter.upper())
        
        ax.add_patch(C)
        ax.axis('equal')
        plt.axis('off')
        plt.ylim((-1.2, 1.2))
        plt.title("Outbeam of '%s' (%d beams)" %(name, len(x)))
        
        return hfig
        
    def plot_inbeam(self, name, visible=True):
        '''Graphic representation of inbeam (see get_outbeam() function above)'''

        letters = list('acehmrz')
        
        theta_letters = []
        for l in letters:
            theta_letters.append(self._letters[l]/(self._nentries+1)*2*np.pi)
        
        x_letters = np.cos(theta_letters)
        y_letters = np.sin(theta_letters)
        
        rank = self._table.get(name, -1)
        
        if rank >= 0: # the word has been found
            rank_i = np.array(self._get_invec(rank))
        else:
            print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
            return {}
        
        theta0 = rank/(self._nentries+1)*2*np.pi
        x0 = np.cos(theta0)
        y0 = np.sin(theta0)
        
        theta = rank_i/(self._nentries+1)*2*np.pi
        x = np.cos(theta)
        y = np.sin(theta)
        
        C = patches.Circle((0,0), 1, edgecolor='k', facecolor=None, fill=False, lw=2)
        
        hfig, ax = plt.subplots()
        for (xk, yk) in zip(x, y):
            plt.plot([x0, xk], [y0, yk], 'k-')

        for (xl, yl, letter) in zip (x_letters, y_letters, letters):
            plt.plot(xl, yl, ls='', marker='o', mec = 'k', mfc = 'k', ms = 7)
            plt.text(1.15*xl, 1.15*yl, letter.upper())
        
        ax.add_patch(C)
        ax.axis('equal')
        plt.axis('off')
        plt.ylim((-1.2, 1.2))
        plt.title("Inbeam of '%s' (%d beams)" %(name, len(x)))
        
    def plot_popularity(self, name=None):
        '''Plots the popularity as a function of the definition length.
        If name is provide, the corresponding entry is also plotted
        as a red cross'''
        
        nFS = self._deflen # Definition length, also card(outbeam)
        nFE = self._quote # Popularity, also card(inbeam)
        
      
        if not name==None:
            rank = self._table.get(name, -1)
            
            if rank >= 0: # the word has been found
                rank_i = np.array(self._get_invec(rank))
            else:
                print("[ERROR] Word '%s' has not been found in the dictionary !" % name)
                return {}            
        
        hfig, ax, = plt.subplots(figsize=(15,15))
        plt.plot(nFS, nFE, 'k+')
        plt.xlabel('Definition length', fontsize=_fontsize)
        plt.ylabel('Popularity', fontsize=_fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Popularity vs. Definition length', fontsize=_fontsize)
        
        if not name==None:
            plt.plot(nFS[rank], nFE[rank], 'r+', mew=3, ms=15)
        
        return hfig
        

if __name__ == '__main__':
    plt.close('all')
    littre = Littre()