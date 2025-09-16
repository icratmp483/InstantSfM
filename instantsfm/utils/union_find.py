
class UnionFind:

    def __init__(self):
        self.parent = {}

    def Find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.Find(self.parent[x])

        return self.parent[x]
    
    def Union(self, x, y):
        root_x = self.Find(x)
        root_y = self.Find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
    
    def Clear(self):
        self.parent.clear()