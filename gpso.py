from PIL import Image, ImageDraw, ImageFilter
from aggdraw import Draw, Brush
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import pickle
import os
import numexpr
import numpy as np


class GPSO(QObject):
    
    update_svg_display = pyqtSignal(str)
    update_iter_indicator = pyqtSignal(int)
    update_n_shapes = pyqtSignal(int)
    update_performance_metrics = pyqtSignal(object)

    
    def __init__(self, **kwargs):
        super().__init__()
        '''
        Args:
            n_population     - int, the number of individuals in the population
            n_shapes         - int, the number of shapes to arrange
            w                - list[float], len=3, weights/probabilities for crossover
            mutation_rate    - list[float], len=6, the mutation rates as described in the paper
            max_iter         - int, maximum number of iterations
            shape_type       - string, 'ellipse' or 'polygon' (only polygon tested)
            n_vert           - int, > 4, the number of vertices for a shape x 2, only for polygons
            output_frequency - int, the frequency of iterations at which to output an image
            bg_color        - string, 'black' or 'white', the image background color
        '''
        self.filename = kwargs.get('filename', None)
        self.directory = kwargs.get('directory', 'mona')
        self.pickle_file = kwargs.get('pickle_file', False)
        
        if self.pickle_file:
            self.load()
            kwargs = self.kwargs
        
        self.n_pop = kwargs.get('n_pop', 200)
        self.max_shapes = kwargs.get('max_shapes', 250)
        self.n_shapes = kwargs.get('n_shapes', 1)
        self.n_vert = kwargs.get('n_vert', 2)
        
        self.shape_type = kwargs.get('shape_type', 'polygon')

        self.x_bits = kwargs.get('x_bits', 6) # x-position bits
        self.y_bits = kwargs.get('y_bits', 6) # y-position bits
        self.c_bits = kwargs.get('c_bits', 4) # Colour bits
        
        if self.shape_type in {'circle', 'square'}:
            self.n_xy = 3
            self.n_xyc = 7
            self.r_bits = min(self.x_bits, self.y_bits)
        elif self.shape_type in {'rectangle', 'ellipse'}:
            self.n_xy = 4
            self.n_xyc = 8
            self.r_bits = 0
        else:
            self.n_xy = self.n_vert * 2
            self.n_xyc = self.n_xy + 4
            self.r_bits = 0

        self.s_bits = self.n_vert \
                    * (self.x_bits + self.y_bits) \
                    + self.r_bits \
                    + self.c_bits * 4
        
        self.w = np.array(kwargs.get('w', [0.65, 0.3, .05]))
        self.w /= np.sum(self.w)
        self.m = kwargs.get('m', [8e-7, 1e-4])
        self.max_iter = kwargs.get('max_iter', 2000)
        self.max_iter = np.inf if self.max_iter == 'inf' else self.max_iter
        self.output_freq = kwargs.get('output_freq', 25)
        self.save_freq = kwargs.get('save_freq', 100)

        self.bg_color = kwargs.get('bg_color', 'black')
        self.color_mode = kwargs.get('image_mode', 'RGB')
        image = Image.open(os.path.normpath(self.filename)) #.convert(self.colour_mode)
        self.image_shape = image.size
        self.reference = np.array(image.getdata(), dtype='uint8')[:,: 3]
        self.reference = (self.reference * np.ones((self.n_pop, *self.reference.shape), dtype='uint8'))
        
        self.pop_range = range(self.n_pop)
        self.log_progress = kwargs.get('log_progress', True)
        self.use_ipython_display = kwargs.get('use_ipython_display', False)
        
        ############################################################################
        # Need to reimplement png mode now that different shape types are possible #
        ############################################################################
#         output_modes = kwargs.get('output_modes', ['svg'])
#         if not output_modes:
#             raise AssertionError('No output_modes')
#         output_mode_translator = {'svg': self.output_svg, 'png': self.output_png}
#         self.output_funcs = [output_mode_translator[mode.lower()] for mode in output_modes]

        self.display_png = kwargs.get('display_png', False)
        self.image_shape = image.size
        self.svg_bg_shape = f'0, {self.image_shape[0]}, 0, {self.image_shape[1]}'
        self.render_shape = np.array(self.image_shape) *np.array([self.n_pop, 1])
        
        bg = '#000000' if self.bg_color == 'black' else '#ffffff'

        self.svg_bg = f'<rect width="{self.image_shape[0]}" height="{self.image_shape[1]}" fill="{bg}" fill-opacity="1"/>\n'
        
        self.initialize_renderers()
        self.initialize_mapping()
        self.initialize_svg_template()
    
        ##############################################################################
        #                          Initialize/Load Population                        # 
        ##############################################################################
        
        # Load previously evolved population                         
        if 'log' in kwargs:
            self.log = kwargs['log']
            self.bin_array = kwargs['bin_array']
            self.p_fit = kwargs['p_fit']
            self.g_fit = kwargs['g_fit']
            self.iter = kwargs['iterations']
            self.max_iter += self.iter
        # Initialize new population    
        else:

            self.log = []
            self.iter = 0
            self.initialize_population()
                                 
        self.kwargs = kwargs
                                 
    def run(self):
                                 
        self.g_flag = False
        i_best = self.max_iter
        
        while 1:
                                 
            self.evaluate(self.reference)
            self.crossover()
            self.mutation()
                                 
            # Record fitness each iteration
            if self.log_progress:
                self.log.append(self.g_fit)
                
            self.generate_svg()
            # Record iteration current best was achieved
            if self.g_flag:
                i_opt = self.iter
                self.g_flag = False
                self.update_svg_display.emit(self.svg)
                
            
#             if self.use_window_display:
#                 self.window_display_svg()
            
            # Output an image to track progress
            if not self.output_freq or self.iter % self.output_freq == 0:
                self.output_svg()
            #     for func in self.output_funcs:
            #         func()
            #     if self.use_ipython_display:
            #         display(SVG(data=self.svg))
                #self.window_display_svg()
                
            
            # Save progress
            # if self.iter % self.save_freq == 0:
            #     self.save()
            
            self.iter += 1
            self.update_iter_indicator.emit(self.iter)
                                 
            # Max iterations
            if self.iter == self.max_iter:
                break
        
        return i_opt, self.g_fit

    def initialize_renderers(self):

        self.renderers = []
        self.bg_brush = Brush((0, 0, 0) if self.bg_color == 'black' else (255, 255, 255), 255)
        self.bg_coords = (0, 0, *self.image_shape)

        if self.shape_type in {'circle', 'square'}:
            self.render_method = self.render_uni_verts
        else:
            self.render_method = self.render_multi_verts

        if self.shape_type == 'circle':
            draw_attr = 'ellipse'
        elif self.shape_type == 'square':
            draw_attr = 'rectangle'
        else:
            draw_attr = self.shape_type

        for _ in self.pop_range:
            draw = Draw('RGBA', self.image_shape, self.bg_color)
            draw.setantialias(False)
            self.renderers.append([draw, getattr(draw, draw_attr)])
    
    def initialize_mapping(self):
        
        ##############################################################################
        #                        Internal operations Mapping                         # 
        ##############################################################################
        
        self.n_bits = self.n_shapes * self.s_bits # bit-length of individual
     
        # Bit_lengths of each dimension in a vector
        # i.e. if x_bits=5, y_bits=7, c_bits=8 and n_verts=3:
        # bit_lens=[5, 7, 5, 7, 5, 7, 8, 8, 8, 8,..., 5, 7, 5, 7, 5, 7, 8, 8, 8, 8]
        if self.shape_type in {'circle', 'square'}:
            bit_lens = np.array([self.x_bits, self.y_bits, self.r_bits, 
                                 self.c_bits, self.c_bits, self.c_bits, self.c_bits], dtype='uint8')
        else:
            bit_lens = (np.append(np.ones(self.n_xy) * np.tile([self.x_bits, self.y_bits], self.n_vert), \
                            [self.c_bits, self.c_bits, self.c_bits, self.c_bits])).astype('uint8')

        # Nested arrays (uneven) for indexing the bits for a given dimension in bin_array
        bit_range = iter(range(self.n_bits))
        self.bin_idx = [[bit_range.__next__() for _ in range(l)] for l in np.tile(bit_lens, self.n_shapes)]

        # For swaping position of shapes in bin_array
        self.swap_idx = np.asarray([[j for i in self.bin_idx[k * self.n_xyc: k * self.n_xyc + self.n_xyc] \
                                       for j in i] for k in range(self.n_shapes)], dtype='int64')
        
        # The base 10 maximums for each dimension in a vector
        div_array = (np.array([int('1' * l, 2) for l in bit_lens]) \
                     * np.ones((self.n_shapes, self.n_xyc))).ravel()
                              
        # For scaling/mapping vectors into solution space
        if self.shape_type in {'circle', 'square'}:
            scale_array = np.tile([*self.image_shape, self.image_shape[0], 255, 255, 255, 255], self.n_shapes) #.astype('uint32')
        else:
            scale_array = np.tile(np.append(np.tile(self.image_shape, self.n_vert),
                                       [255, 255, 255, 255]), self.n_shapes) #.astype('uint32')

        # For mapping from search space to solution space                        
        self.map = scale_array / div_array
                                 
        # For indexing shapes in individuals 
        self.vec_idx = np.split(np.arange(self.n_shapes * (self.n_xyc)),
                             np.cumsum([self.n_xyc] * self.n_shapes))[:-1]
        
        self.static_idx = np.ogrid[: self.n_pop, : self.n_bits] # Used for indexing the 3D binary array during crossover

        if self.shape_type in {'circle', 'square'}:
            # The vec_array positions that will become the coordinates (x, y) for the shape centres
            #                             c1,c2,r1,r2, r, g, b, a
            self.uni_vert_mask = np.tile([0, 0, 0, 1, 0, 0, 0, 0], self.n_shapes).astype(bool)

            self.x1_mask = np.tile([1, 0, 0, 0, 0, 0, 0], self.n_shapes).astype(bool)
            self.x2_mask = np.tile([0, 0, 1, 0, 0, 0, 0], self.n_shapes).astype(bool)
            self.y1_mask = np.tile([0, 1, 0, 0, 0, 0, 0], self.n_shapes).astype(bool)

                        # The vec_array positions that will become the coordinates (x, y) for the shape centres
            #                     c1,c2,r1, r, g, b, a
            self.c_mask = np.tile([1, 1, 0, 0, 0, 0, 0, 0], self.n_shapes).astype(bool)
            # The vec_array positions that will become the radii in the x, y directions
            self.r_mask = np.tile([0, 0, 1, 1, 0, 0, 0, 0], self.n_shapes).astype(bool)

        elif self.shape_type in {'ellipse', 'rectangle'}:
            # The vec_array positions that will become the coordinates (x, y) for the shape centres
            #                     c1,c2,r1,r2, r, g, b, a
            self.c_mask = np.tile([1, 1, 0, 0, 0, 0, 0, 0], self.n_shapes).astype(bool)
            #self.c2_mask = np.tile([0, 1, 0, 0, 0, 0, 0, 0], self.n_shapes).astype(bool)
            # The vec_array positions that will become the radii in the x, y directions
            self.r_mask = np.tile([0, 0, 1, 1, 0, 0, 0, 0], self.n_shapes).astype(bool)
            # Both of the above
            self.cr_mask = self.c_mask | self.r_mask

        if self.shape_type in {'circle', 'square'}:
            self.rgb_mask = np.tile([0, 0, 0, 0, 1, 1, 1, 0], self.n_shapes).astype(bool) # RGB values
            self.a_mask = np.tile([0, 0, 0, 0, 0, 0, 0, 1], self.n_shapes).astype(bool) # Alpha values
        else:
            self.rgb_mask = np.tile(np.hstack([np.zeros(self.n_xy), [1, 1, 1, 0]]), self.n_shapes).astype(bool) # RGB values
            self.a_mask = np.tile(np.hstack([np.zeros(self.n_xy), [0, 0, 0, 1]]), self.n_shapes).astype(bool) # Alpha values

    def initialize_svg_template(self):

        # if self.shape_type == 'circle':
        #      self.shape_template = '<circle cx="%f" cy="%f" r="%f" fill="#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n'

        if self.shape_type in {'circle', 'ellipse'}:
            self.shape_template = '<ellipse cx="%f" cy="%f" rx="%f" ry="%f" fill="#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n'

        elif self.shape_type in {'square', 'rectangle'}:
            self.shape_template = '<rect x="%f" y="%f" width="%f" height="%f" fill="#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n'

        elif self.shape_type == 'polygon':
            self.shape_template = ('<polygon points="%s" fill="@' % ''.join('%f,' for _ in range(self.n_xy))).replace('@', '#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n')

        self.svg_template = '<svg viewBox="0 0 %s %s" xmlns="http://www.w3.org/2000/svg">\n' % self.image_shape + self.svg_bg + ''.join(self.shape_template for _ in range(self.n_shapes))

    def initialize_population(self):
        '''
        Initialize a population at random positions in the search space
        '''                        
        # Binary representations
        self.bin_array = np.random.randint(2, size=(self.n_pop, self.n_bits))
        
        # Bin_array is a 3D array which will contain the population, personal best and global best
        # in binary form. We first take 3 copies of bin_array to create the correct shape and then
        # replace as we go 
        self.bin_array = np.array([self.bin_array, self.bin_array, self.bin_array], dtype='uint8')

        # Initialize p and g fitness as inf
        self.p_fit = np.full(self.n_pop, np.inf)
        self.g_fit = np.inf

    def crossover(self):
        '''
        Crossover is performed by some indexing of the 3D binary array
        '''
        #mask = choice(3, size=(self.n_pop, self.n_bits), p=self.w)
        self.bin_array[0] = self.bin_array[
                                           np.random.choice(3, size=(self.n_pop, self.n_bits), p=self.w),
                                           self.static_idx[0],
                                           self.static_idx[1]
                                           ]
    def mutation(self):
        
        self.flip_bits()
        
#         if rand() < self.m[1]:
#             self.initialize_individual()
        
        if self.n_shapes < self.max_shapes \
        and self.iter > 1 \
        and self.iter % 650 == 0:
            self.increase_n_shapes()
        
        if self.n_shapes > 1 and np.random.rand() < self.m[3]:
            self.swap_indices()
        
        #self.initialize_individual()
        
    def flip_bits(self):
        '''
        Performs some mutation on the population
        '''
        mutants = np.random.rand(self.n_pop, self.n_bits) < self.m[0]

        mutant_0s = (mutants) & (self.bin_array[0] == 0.)
        mutant_1s = (mutants) & (self.bin_array[0] == 1.)

        self.bin_array[0, mutant_0s] = 1.
        self.bin_array[0, mutant_1s] = 0.
        
    def increase_n_shapes(self):
        self.n_shapes += 1
        self.initialize_mapping()
        self.svg_template += self.shape_template
        self.bin_array = np.dstack([self.bin_array, np.random.randint(2, size=(3, self.n_pop, self.s_bits))])
        self.bin_to_vec()
        self.s_flag = True
        self.update_n_shapes.emit(self.n_shapes)
        
    def swap_indices(self):
        mutant = np.random.randint(self.n_pop)
        self.bin_array[0, mutant, :] = self.bin_array[0, mutant, np.random.permutation(self.swap_idx).ravel()]
        
    def initialize_individual(self):
        self.w_idx = np.argmax(self.fitness)
        self.bin_array[0][self.w_idx] = np.random.randint(2, size=(self.n_bits))
                                 
    def evaluate(self, reference):
        '''
        Evaluate fitness and update bests
        ''' 
        self.bin_to_vec()
        self.vec_array = self.map_to_space(self.vec_array)

        imgs = np.array([self.render_images(i) for i in self.pop_range], dtype='int64')
        

        fitness = numexpr.evaluate('sum((imgs - reference) ** 2.0, 1)', order='C')
        self.fitness = numexpr.evaluate('sum(fitness, 1)', order='C')
        # Bool array of personal best locations
        p_idx = self.fitness < self.p_fit
        # Find all indices equal to the best and select one at random
        self.g_idx = np.random.choice(np.ravel(np.where(self.fitness==self.fitness[np.argmin(self.fitness)])))
                       
        if self.fitness[self.g_idx] < self.g_fit or self.s_flag:
            # The global best is copied so that it is the same shape as the population array
            self.bin_array[2] = (self.bin_array[0][self.g_idx] * np.ones((self.n_pop, self.n_bits))).astype('uint8')
            self.g_fit = self.fitness[self.g_idx]
            # record new best achieved
            self.g_best = imgs[self.g_idx]
            self.g_flag = True
            self.s_flag = False
            
        # Update personal bests
        self.bin_array[1][p_idx] = self.bin_array[0][p_idx]
        self.p_fit[p_idx] = self.fitness[p_idx]

        # Calculate current metrics for plotting purposes
        self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        distance = np.sqrt(self.fitness)
        bst = distance[self.g_idx]
        avg = np.mean(distance)
        std = np.std(distance)
        std_hi = avg + std
        std_lo = avg - std
        wst = distance[np.argmax(self.fitness)]
        self.update_performance_metrics.emit([self.iter, [wst], [std_hi, std_lo], [avg], [bst]])

    def base2_to_base10(self, x):
        '''
        For converting a binary array into a base 10 integer
        '''
        o = np.zeros(x.shape[1], dtype='int64')
        for bits in x:
            o = (o << 1) | bits
        return o

    def bin_to_vec(self):
        '''
        Converts the binary array into an array of vectors
        '''
        self.vec_array = np.array([self.base2_to_base10(self.bin_array[0, :, i]) for i in self.bin_idx]).T
        
    def map_to_space(self, x):
        return (x * self.map).round(0).astype('int64')

    def translate_circles(self, x):
        # Calculate which x2 need clipping

        x[2::7] = (x[2::7] - x[::7]) / 2.
        # Duplicate x2 -> r1, r2
        o = np.empty(x.shape[0] + self.n_shapes, dtype='float64')
        o[~self.uni_vert_mask ] = x
        o[self.uni_vert_mask ] = x[2::7]
        # Centres
        o[self.c_mask] += o[self.r_mask]
        return o

    def translate_squares(self, x):
        # Find x and y where x1 > x2 or y1 > y2
        idx = x[self.x1_mask] < x[self.x2_mask]
        # Create temporary intermediate boolean arrays
        temp1, temp2 = self.x1_mask.copy(), self.x2_mask.copy()
        # Switch xs and/or ys which match requirements
        temp1[self.x2_mask == True] = idx
        temp2[self.x1_mask == True] = idx
        x[temp1], x[temp2] = x[temp2], x[temp1]

        # Calculate diameter/width
        x[2::7] = x[2::7] - x[::7]

        # Duplicate x2 -> r1, r2
        o = np.empty(x.shape[0] + self.n_shapes, dtype='float64')
        o[~self.uni_vert_mask ] = x
        o[self.uni_vert_mask ] = x[2::7]
        
        return o


    def translate_ellipses(self, x): 
        '''
        The SVG standard requires an ellipse to be represented by its centre coordinates and radii; which
        differs from that of aggdraw where they are represented by the bottom-left and upper-right coordinates
        of its rectangular bounding box

        Input:
            x - array-like, shape=(1, -1), individual solution encoded as a vector
        
        Output:
            x - array-like, shape=(1, n_shapes * (n_vert + 4)), example: [cx1, cy2, rx1, ry2, r, g, b, a,...]
                where cxi, cyi = centre and rxi, ryi = radii
        '''
        radii = (x[self.r_mask] - x[self.c_mask]) / 2.

        x[self.c_mask] += radii # Ellipse centres
        x[self.r_mask] = radii  # Ellipse radii

        return x

    def translate_rectangles(self, x): 
        '''
        The SVG standard requires a rectangle to be represented by the x, y coordinates of its upper-left
        corner and its width and height

        Input:
            x - array-like, shape=(1, -1), individual solution encoded as a vector
        
        Output:
            x - array-like, shape=(1, n_shapes * (n_vert + 4)), example: [cx1, cy2, wx1, hy2, r, g, b, a,...]
                where cxi, cyi = centre and width and height = d
        '''
        # Find x and y where x1 > x2 or y1 > y2
        idx = x[self.c_mask] > x[self.r_mask]
        # Create temporary intermediate boolean arrays
        temp1, temp2 = self.r_mask.copy(), self.c_mask.copy()
        # Switch xs and/or ys which match requirements
        temp1[self.r_mask == True] = idx
        temp2[self.c_mask == True] = idx
        x[temp1], x[temp2] = x[temp2], x[temp1]
        # Calculate widths and heights
        x[self.r_mask] -= x[self.c_mask]
        return x

    def generate_svg(self):
        g_best = self.vec_array[self.g_idx].astype('float64')
        # SVG standard (non-CSS) expects alpha to be float (0 <= x <= 1)
        if self.shape_type == 'circle':
            g_best = self.translate_circles(g_best)
        elif self.shape_type == 'square':
            g_best = self.translate_squares(g_best)
        elif self.shape_type == 'ellipse':
            g_best = self.translate_ellipses(g_best)
        elif self.shape_type == 'rectangle':
            g_best = self.translate_rectangles(g_best)

        g_best[self.a_mask] /= 255.
        svg = self.svg_template.format(*g_best[self.rgb_mask].astype('uint8'))
        svg = svg % tuple(g_best[~self.rgb_mask])
        svg += '</svg>'
        self.svg = svg
        #print(svg)
        
    def output_svg(self):
        with open(os.path.normpath(f'{self.directory}/{self.iter}.svg'), 'w') as file:
            file.write(self.svg)

    def output_png(self):
        '''
        Render and output an image
        '''
        image = self.vec_array[self.g_idx]
        
        if self.use_output_mapping:
            image = (image * self.output_map).round(0).astype(uint8)[0]
            draw = Draw('RGBA', self.output_shape, self.bg_color)
        else:
            draw = Draw('RGBA', self.image_shape, self.bg_color)
        
        if self.shape_type == 'ellipse':
            for idx in self.vec_idx:
                draw.ellipse(image[idx][: self.n_xy], 
                             Brush(tuple(image[idx][self.n_xy: -1]), 
                                   image[idx][-1]))
        else:
            for idx in self.vec_idx:
                draw.polygon(image[idx][: self.n_xy], 
                             Brush(tuple(image[idx][self.n_xy: -1]), 
                                   image[idx][-1]))
        Image.frombytes('RGBA', 
                        self.output_shape, 
                        draw.tobytes()).save(os.path.normpath(f'{self.directory}/n_{self.iter}.png'))
        
        #if self.display_png:
        #    imshow(image)
        #    show()

    def render_uni_verts(self, i, shape):
        '''
        For rendering shapes that have a single vertex and a radius such as circles and squares
        '''
        self.renderers[i][1]((*shape[: self.n_xy], shape[1] + shape[2] - shape[0]), Brush(tuple(shape[self.n_xy: -1]), shape[-1]))

    def render_multi_verts(self, i, shape):
        '''
        For rendering shapes that have multiple vertices: rectangles and polygons
        '''
        self.renderers[i][1](shape[: self.n_xy], Brush(tuple(shape[self.n_xy: -1]), shape[-1]))

    def render_images(self, i: int):
        '''
        Converts an individual into an image 
        '''
        self.renderers[i][0].rectangle(self.bg_coords, self.bg_brush)
        for idx in self.vec_idx:
            self.render_method(i, self.vec_array[i, idx])

        return np.frombuffer(self.renderers[i][0].tobytes(), dtype='uint8').reshape(-1, 4, order='C')[:,:3]

    def save(self):
        '''
        Save this instance
        '''
        with open(os.path.normpath(f'{self.directory}/{self.filename[: -4]}_pickle'), 'wb') as open_file:
            self.kwargs['bin_array'] = self.bin_array
            self.kwargs['p_fit'] = self.p_fit
            self.kwargs['g_fit'] = self.g_fit
            self.kwargs['iterations'] = self.iter
            self.kwargs['log'] = self.log
            self.kwargs['n_shapes'] = self.n_shapes
            pickle.dump(self.kwargs, open_file)
        
    def load(self):
        '''
        Load a previous instance
        '''
        with open(os.path.normpath('%s/%s' % (self.directory, self.pickle_file)), 'rb') as open_file:
        
            self.kwargs = pickle.load(open_file)