import numpy as np

class Node(object): # 单个节点
    def __init__(self, data = None, lchild = None, rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild

class KDNode(Node): # axis for father axis  sel_axis for now
    def __init__(self, data = None, lchild = None,rchild = None ,axis = None,sel_axis=None,dimensions=None):
        super(KDNode, self).__init__(data,lchild,rchild)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions


    def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
        if not point_list and not dimensions:
            raise ValueError('either point_list or dimensions should be provided')
        elif point_list:
            dimensions = check_dimensionality(point_list, dimensions)

        sel_axis = sel_axis or (lambda prev_axis:(prev_axis+1)%dimensions)

        # empty tree
        if not point_list:
            return KDNode(sel_axis=sel_axis,axis = axis,dimensions = dimensions)

        point_list = list(point_list)

        # 在指定的维度上排序
        point_list.sort(key=lambda point:point[axis])
        median = len(point_list)/2

        loc = point_list[median]
        lchild = create(point_list[:median], dimensions, axis, sel_axis)
        rchild = create(point_list[median+1:], dimensions, axis, sel_axis)
        return KDNode(loc ,lchild, rchild, axis, sel_axis,dimensions)


    def check_dimensionality(point_list, dimensions=None):

        dimensions = dimensions or len(point_list[0])
        for p in point_list:
            if len(p) != dimensions:
                raise ValueError('All Points in the point_list must have the same dimensionality')
        return dimensions

    # 用一个优先队列来保存最近的k个实例

    def _search_node(self,point, k, results, get_dist):
        if not self:
            return
        nodeDist = get_dist(self)

        # priority queue
        # 如果当前的节点小于队列中的任何一个节点,则将该节点添加入队列中
        # 该功能的实现在BoundPriorityQueue类中实现
        results.add((self, nodeDist))


        # 获取当前节点的切分平面
        split_plane = self.data[self.axis]
        plane_dist = point[self.axis]
        plane_dist2 = plane_dist **2


        # 从从根节点递归向下查找
        if point(self.axis) < split_plane:
            if self.lchild is not None:
                self.lchild._search_node(point,k,results,get_dist)
        else:
            if self.rchild is not None:
                self.rchild._search_node(point,k,results,get_dist)



        # 回溯查找
        # 第一个条件指的是特征圆的判断,也就是同一个夫结点的另外一个分支是否存在可能的解
        # 第二个条件指的是获取初始的k个值
        if plane_dist2 < results.max() or results.size() < k:
            if pointp[self.axis] < self.data[self.axis]:
                if self.rchild is not None:
                    self.rchild._search_node(point,k,results,get_dist)
            else:
                if self.lchild is not None:
                    self.lchild._search_node(point,k,results,get_dist)

class BoundedPriorityQueue:
    """优先队列(max heap)及相关实现函数"""

    def __init__(self, k):
        self.heap = []
        self.k = k

    def items(self):
        return self.heap

    def parent(self, index):
        """返回父节点的index"""
        return int(index / 2)

    def left_child(self, index):
        return 2 * index + 1

    def right_index(self, index):
        return 2 * index + 2

    def _dist(self, index):
        """返回index对应的距离"""
        return self.heap[index][3]

    def max_heapify(self, index):
        """
        负责维护最大堆的属性，即使当前节点的所有子节点值均小于该父节点
        """
        left_index = self.left_child(index)
        right_index = self.right_index(index)

        largest = index
        if left_index < len(self.heap) and self._dist(left_index) > self._dist(index):
            largest = left_index
        if right_index < len(self.heap) and self._dist(right_index) > self._dist(largest):
            largest = right_index
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self.max_heapify(largest)

    def propagate_up(self, index):
        """在index位置添加新元素后，通过不断和父节点比较并交换
            维持最大堆的特性，即保持堆中父节点的值永远大于子节点"""
        while index != 0 and self._dist(self.parent(index)) < self._dist(index):
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[
                index]
            index = self.parent(index)

    def add(self, obj):
        """
        如果当前值小于优先队列中的最大值，则将obj添加入队列，
        如果队列已满，则移除最大值再添加，这时原队列中的最大值、
        将被obj取代
        """
        size = self.size()
        if size == self.k:
            max_elem = self.max()
            if obj[1] < max_elem:
                self.extract_max()
                self.heap_append(obj)
        else:
            self.heap_append(obj)

    def heap_append(self, obj):
        """向队列中添加一个obj"""
        self.heap.append(obj)
        self.propagate_up(self.size() - 1)

    def size(self):
        return len(self.heap)

    def max(self):
        return self.heap[0][4]

    def extract_max(self):
        """
        将最大值从队列中移除，同时从新对队列排序
        """
        max = self.heap[0]
        data = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = data
            self.max_heapify(0)
        return max

    def search_knn(self, point, k, dist=None):
        """返回k个离point最近的点及它们的距离"""

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            gen_dist = lambda n: dist(n.data, point)

        results = BoundedPriorityQueue(k)
        self._search_node(point, k, results, get_dist)

        # 将最后的结果按照距离排序
        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)

