# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 2018
@author: xwy
@environment: python2.7
"""

class Node:
  def __init__(self, freq, symbol):
    self.left = None
    self.right = None
    self.father = None
    self.code = ''
    self.freq = freq
    self.symbol = symbol
  def isLeft(self):
    return self.father.left == self
  def isLeaf(self):
    return (self.left == None and self.right == None)

def createNodes(freqs):
  return [Node(freq, symbol) for symbol, freq in freqs.items()]

def createHuffmanTree(nodes):
  queue = nodes[:]
  while len(queue) > 1:
    queue.sort(key=lambda item:item.freq)
    node_left = queue.pop(0)
    node_right = queue.pop(0)
    node_father = Node(node_left.freq + node_right.freq, '')
    node_father.left = node_left
    node_father.right = node_right
    node_left.father = node_father
    node_right.father = node_father
    queue.append(node_father)
  queue[0].father = None
  return queue[0]

def dfs(node):
  if node != None:
    if node.isLeft():
      node.code = node.father.code + '1'
    else:
      node.code = node.father.code + '0'
    dfs(node.left)
    dfs(node.right)

def HuffmanEncoding(root):
  if (root != None):
    dfs(root.left)
    dfs(root.right)


def getHuffmanCode(freqs, HuffmanTable):
  nodes = createNodes(freqs)
  root = createHuffmanTree(nodes)
  HuffmanEncoding(root)
  for node in nodes:
    if node.symbol not in HuffmanTable:
      HuffmanTable[node.symbol] = node.code
    # print node.symbol, node.freq, node.code