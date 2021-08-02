import collections
import numpy as np
Object = collections.namedtuple('Object', ['id', 'score', 'bbox']) class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])): """Bounding box. Represents a rectangle which sides are either vertical or horizontal, parallel to the x or y axis. """ __slots__ = () @ property def width(self):
