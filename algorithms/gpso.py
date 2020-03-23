import sys
import os
import pickle
import time

from PIL import Image, ImageDraw, ImageFilter
from aggdraw import Draw, Brush
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication
import numexpr
import numpy as np


class GPSO(QObject):

    id_ = 'gpso'

    display_signal             = pyqtSignal(str)
    iter_indicators_signal     = pyqtSignal(object)
    fitness_indicator_signal   = pyqtSignal(float)
    n_shape_signal             = pyqtSignal(int)
    performance_metrics_signal = pyqtSignal(object)
    status_signal              = pyqtSignal(str)
    
    pause_icon_path            = os.path.join(os.path.split(os.path.abspath(__file__))[0], os.path.pardir, 'img', 'pause_icon.svg')

    population_b10             = None
    prev_population_b2         = None
    svg                        = None
    bgs                        = None
    save_flag                  = False
    pause_flag                 = False
    g_flag                     = False
    new_shape_flag             = False
    reinit_shape_flag          = False
    end_flag                   = False
    iter                       = 0
    iter_with_no_improvement              = 0
    thread_enabled             = True

    # Attributes that get stored when progress is saved
    to_backup = [
        'id_',
        'img_path', 'img_name', 'output_dir', 'output_freq', 'save_freq', 'n_pop', 'n_shape', 'max_shapes', 'n_vert', 'shape_type',
        'w', 'm', 'img_size', 'reference', 'draw_method',
        'x_bits', 'y_bits', 'c_bits', 'n_xy', 'n_xyc', 'r_bits', 's_bits',
        'translation_ref', 'bg_color', 'pop_range',
        'shape_management_ref', 'shape_management_prob', 'shape_management_delta', 'shape_management_interval', 
        'oaat_mode', 'rollback_mode', 

        'svg_bg_shape', #'render_shape', 
        'svg_bg',
        'iter', 'g_flag', 'new_shape_flag', 'reinit_shape_flag', 'iter_with_no_improvement',

        'population_b2', 'p_dist', 'g_dist', 'g_idx',

        'prev_population_b2', 'prev_g_dist',
    ]

    def __init__(self, **kwargs):
        super().__init__()

        '''
        Args:
            n_pop                   - int, the number of individuals in the population
            n_shape                 - int, the number of shapes to arrange
            max_shapes              - int, the maximum number of shapes that can be grown/evolved by 
                                      algorithm
            shape_type              - string, 'ellipse' or 'polygon' (only polygon tested)
            n_vert                  - int, > 4, the number of vertices for a shape x 2, only for polygons
            shape_management_method   - str, 'probabilistic', 'velocity', 'periodic'
            shape_management_prob     - *float, probability that a new shape will be generated
            shape_management_interval - *float, number of iterations to occur before a new shape is generated
            shape_management_delta    - *float, number of iterations with no improvement before a new shape
                                      is generated
                                      ***depending on which shape_management_method is selected
            w                       - list[float], len=3, weights/probabilities for crossover
            m                       - list[float], len=6, the mutation rates, [bit_flip, shape_swap]
            img_path                - str, the path to the image which we are going to evolve
            output_dir              - str, path to a folder where we will dump SVG images
            progress_dir            - str, where to save the population, default=None, if a path is given
                                      a previously instantiated population will be loaded, else the algorithm
                                      will start fresh
            save_freq               - int, how often (in iterations) the optimization progress is saved and 
                                      the population stored in progress_dir
            output_freq             - int, how often (in iterations) the optimizer outputs an SVG image in 
                                      output_dir
            x_bits                  - int, the number of bits used to represent the x-dimension of the image
            y_bits                  - int, the number of bits used to represent the y-dimension of the image
            oaat_mode               - bool, whether or not to use the One At A Time version of the algorithm
                                      which basically optimizes a single shape at a time. This approach is
                                      much faster but sacrifices some fitness of the solutions obtained
            load_flag               - bool, whether or not to load a previously instantiated population
            bg_color                - string, 'black' or 'white', the image background color

        Constructed attributes:
            n_bits                  - int, the number of bits that encode a single base 2 solution
            s_bits                  - int, the number of bits that encode a single shape
            population_b2           - array-like, shape=(3, n_pop, n_bits), the first n_pop x n_bits shaped 
                                      2D array (population_b2[0]) contains the binary encoding of each individual
                                      in the population. The second 2D array (population_b2[1]) contains the 
                                      personal best solutions  obtained by the corresponding individuals in 
                                      population_b2[0]. The 3rd array (population_b2[2]) contains the single global 
                                      best solution copied n_pop times.
            population_b10          - array-like, shape=(n_pop, n_xyc * n_shape), a 2D array containing the
                                      base 10 encoded current solution/position of each individual.
        '''

        if not kwargs.get('load_flag', False): # Are we going to start fresh?
            self._init(**kwargs)
        else:                                  # Or load a previously saved population?
            self._reinit(**kwargs)
        
        self.initialize_renderers()
        self.initialize_svg_template()

        self.load_pause_icon()
        self.parent_display      = kwargs['parent_display']
        kwargs['parent_display'] = None
        self.parent_display.button_map['Save state'].connect(self.toggle_save_flag)
        self.parent_display.button_map['Pause'].connect(self.toggle_pause_flag)
        # Store kwargs as an attribute for saving/loading progress later
        self.kwargs = kwargs

    def _init(self, **kwargs):

        self.img_path      = kwargs['img_path']
        path_components    = os.path.split(self.img_path)
        self.img_name      = path_components[-1].split('.')[0]
        self.output_dir    = kwargs.get('output_dir', os.path.join(*path_components)[:-1])
        self.progress_path = os.path.join(kwargs.get('progress_dir', self.output_dir), self.img_name + '_progress.pkl')

        self.output_freq = kwargs.get('output_freq', 25)
        self.save_freq   = kwargs.get('save_freq', 100)

        self.n_pop      = kwargs.get('n_pop', 128)
        self.n_shape    = kwargs.get('n_shape', 1)
        self.max_shapes = kwargs.get('max_shapes', 64)
        self.n_vert     = kwargs.get('n_vert', 2)
        self.shape_type = kwargs.get('shape_type', 'polygon')

        self.w  = np.array(kwargs.get('w', [0.1, 0.45, 0.45]))
        self.w /= np.sum(self.w)
        self.m  = kwargs.get('m', [1e-5, 1e-4])

        img = Image.open(os.path.normpath(self.img_path))
        self.img_size  = img.size
        self.reference = np.array(img.getdata(), dtype='uint8')[:, : 3]
        self.reference = (self.reference * np.ones((self.n_pop, *self.reference.shape), dtype='int64'))

        self.x_bits = kwargs.get('x_bits', 6)  # x-position bits
        self.y_bits = kwargs.get('y_bits', 6)  # y-position bits
        self.c_bits = kwargs.get('c_bits', 4)  # Colour bits

        if self.shape_type == 'polygon':
            self.n_xy   = self.n_vert * 2
            self.n_xyc  = self.n_xy + 4
            self.r_bits = 0
            # Translation methods adapt geometry between SVG and aggdraw
            self.translation_ref = None
        else:
            self.translation_ref = 'translate_' + self.shape_type + 's'
            if self.shape_type in {'circle', 'square'}:
                self.n_xy   = 3
                self.n_xyc  = 7
                self.r_bits = min(self.x_bits, self.y_bits)
            elif self.shape_type in {'rectangle', 'ellipse'}:
                self.n_xy   = 4
                self.n_xyc  = 8
                self.r_bits = 0

        # The number of bits per shape
        self.s_bits = self.n_vert \
                      * (self.x_bits + self.y_bits) \
                      + self.r_bits \
                      + self.c_bits * 4

        self.pop_range = range(self.n_pop)

        self.shape_management_ref      = kwargs.get('shape_management_method', 'velocity') + '_shape_management'
        self.shape_management_method   = getattr(self,  self.shape_management_ref)
        self.shape_management_prob     = kwargs.get('shape_management_prob', 0.05)
        self.shape_management_delta    = kwargs.get('shape_management_delta', 2000)
        self.shape_management_interval = kwargs.get('shape_management_interval', 100)

        self.translation_method = getattr(self, self.translation_ref) if self.translation_ref else None
        self.oaat_mode          = kwargs.get('oaat_mode', False)
        self.rollback_mode      = kwargs.get('rollback_mode', True)

        if self.oaat_mode:
            self.evaluate   = self.evaluate_oaat_mode
            self.crossover  = self.crossover_oaat_mode
            self.mutation   = self.mutation_oaat_mode

        self.bg_color = kwargs.get('bg_color', 'black').lower()
        self.svg_bg_shape = f'0, {self.img_size[0]}, 0, {self.img_size[1]}'

        # Background for svg
        bg_hex      = '#000000' if self.bg_color == 'black' else '#ffffff'
        self.svg_bg = f'<rect width="{self.img_size[0]}" height="{self.img_size[1]}" fill="{bg_hex}" fill-opacity="1"/>\n'

        self.initialize_mapping()
        self.initialize_population()

    def _reinit(self, **kwargs):
        self.progress_path = kwargs['progress_dir']
        self.load()
        self.unpack()
        if kwargs['output_dir']:
            self.output_dir = kwargs['output_dir']
        self.translation_method = getattr(
            self, self.translation_ref) if self.translation_ref else None
        if self.oaat_mode:
            self.evaluate  = self.evaluate_oaat_mode
            self.crossover = self.crossover_oaat_mode
            self.mutation  = self.mutation_oaat_mode
        self.shape_management_method = getattr(self, self.shape_management_ref)
        self.initialize_mapping()

    def backup(self):
        for attr in self.to_backup:
            self.kwargs[attr] = getattr(self, attr)

    def unpack(self):
        for attr in self.to_backup:
            try:
                setattr(self, attr, self.kwargs[attr])
            except KeyError:
                setattr(self, attr, None)

    def load(self):
        '''
        Load a previous instance
        '''
        with open(self.progress_path, 'rb') as open_file:
            self.kwargs = pickle.load(open_file)

    def save(self):
        '''
        Save this instance
        '''
        try:
            with open(self.progress_path, 'wb') as open_file:
                pickle.dump(self.kwargs, open_file)
            self.status_signal.emit('Saved')
        except PermissionError:
            self.status_signal.emit('Save failed, another process may be accessing the file. Try again later.')

    def load_pause_icon(self):
        with open(self.pause_icon_path, 'r') as open_file:
            self.pause_icon = ''.join(open_file.readlines())

    @pyqtSlot()
    def toggle_save_flag(self):
        self.save_flag = True

    @pyqtSlot()
    def toggle_pause_flag(self):
        self.pause_flag = not self.pause_flag

    def pause(self):
        self.display_signal.emit(self.pause_icon)
        self.status_signal.emit('Paused')
        while 1:
            if not self.pause_flag:
                break
            #time.sleep(0.03)
            QApplication.processEvents()
        self.status_signal.emit('Unpaused')
        self.display_signal.emit(self.svg)

    def run(self):
        '''
        This is the main loop of the algorithm
        '''

        while self.thread_enabled:

            self.evaluate()

            # New global best achieved? generate SVG image...
            if self.g_flag or self.svg is None:
                self.generate_svg()
                self.display_signal.emit(self.svg)
                self.g_flag = False

            # Output an image to track progress
            if not self.output_freq or self.iter % self.output_freq == 0:
                self.output_svg()

            # Increase number of shapes
            if self.n_shape < self.max_shapes and self.iter > 1 and self.shape_management_method():
                self._shape_management()
            else:
                self.crossover()
                self.mutation()

            self.iter += 1
            self.iter_indicators_signal.emit([self.iter, self.iter_with_no_improvement])

            # Save progress
            if self.save_freq and self.iter % self.save_freq == 0 \
                or self.save_flag:
                self.backup()
                self.save()
                self.save_flag = False

            if self.pause_flag:
                self.pause()

            if self.end_flag:
                self.quit()

            QApplication.processEvents()

    def initialize_renderers(self):
        '''
        We will initialize the draw objects for each individual/solution as to save computations.
        Instead of reinitializing them each iteration we will wipe the images by drawing a
        opaque rectangle in white or black. Then render the shapes on top. This method creates
        those draw objects and stores them in a list.
        '''
        self.draws = []
        self.bg_brush = Brush((0, 0, 0) if self.bg_color == 'black' else (255, 255, 255), 255)
        self.bg_coords = (0, 0, *self.img_size)

        if self.shape_type in {'circle', 'square'}:
            self.render_method = self.render_uni_verts
        else:
            self.render_method = self.render_multi_verts

        if self.shape_type   == 'circle':
            self.draw_method = 'ellipse'
        elif self.shape_type == 'square':
            self.draw_method = 'rectangle'
        else:
            self.draw_method = self.shape_type

        for _ in self.pop_range:
            draw = Draw('RGBA', self.img_size, self.bg_color)
            draw.setantialias(False)
            self.draws.append([draw, getattr(draw, self.draw_method)])

    def initialize_mapping(self):
        '''
        This method initializes some objects which are used for indexing arrays in a complex manner
        '''
        ##############################################################################
        #                        Internal operations Mapping                         #
        ##############################################################################

        self.n_bits = self.n_shape * self.s_bits  # bit-length of individual

        # Bit_lengths of each dimension in a vector
        # i.e. if x_bits=5, y_bits=7, c_bits=8 and n_verts=3:
        # bit_lens=[5, 7, 5, 7, 5, 7, 8, 8, 8, 8,..., 5, 7, 5, 7, 5, 7, 8, 8, 8, 8]
        if self.shape_type in {'circle', 'square'}:
            bit_lens = np.array([self.x_bits, self.y_bits, self.r_bits,
                                 self.c_bits, self.c_bits, self.c_bits, self.c_bits], dtype='uint8')
        else:
            bit_lens = (np.append(np.ones(self.n_xy) * np.tile([self.x_bits, self.y_bits], self.n_vert),
                                     [self.c_bits, self.c_bits, self.c_bits, self.c_bits])).astype('uint8')

        # Nested arrays (uneven) for indexing the bits for a given dimension in population_b2
        b2_iter = iter(range(self.n_bits))
        self.b2_idx = [[b2_iter.__next__() for _ in range(l)] for l in np.tile(bit_lens, self.n_shape)]

        # For swaping position of shapes in population_b2
        self.swap_idx = np.asarray([[j for i in self.b2_idx[k * self.n_xyc: k * self.n_xyc + self.n_xyc]
                                       for j in i] for k in range(self.n_shape)], dtype='int64')

        # The base 10 maximums for each dimension in a vector
        div_array = (np.array([int('1' * l, 2) for l in bit_lens]) * np.ones((self.n_shape, self.n_xyc))).ravel()

        # For scaling/mapping vectors into solution space
        if self.shape_type in {'circle', 'square'}:
            scale_array = np.tile([*self.img_size, self.img_size[0], 255, 255, 255, 255], self.n_shape)  # .astype('uint32')
        else:
            scale_array = np.tile(np.append(np.tile(self.img_size, self.n_vert), [255, 255, 255, 255]), self.n_shape)  # .astype('uint32')

        # For mapping from search space to solution space
        self.map = (scale_array / div_array).astype('float64')

        # For indexing shapes in individuals
        self.b10_idx = np.split(np.arange(self.n_shape * (self.n_xyc)), np.cumsum([self.n_xyc] * self.n_shape))[:-1]

        # Used for indexing the 3D binary array during crossover
        if self.oaat_mode:
            self.static_idx = np.ogrid[: self.n_pop, : self.s_bits]
        else:
            self.static_idx = np.ogrid[: self.n_pop, : self.n_bits]

        if self.shape_type in {'circle', 'square'}:
            # The population_b10 positions that will become the coordinates (x, y) for the shape centres
            #                             c1,c2,r1,r2, r, g, b, a
            self.uni_vert_mask = np.tile([0, 0, 0, 1, 0, 0, 0, 0], self.n_shape).astype('bool')

            self.x1_mask = np.tile([1, 0, 0, 0, 0, 0, 0], self.n_shape).astype('bool')
            self.x2_mask = np.tile([0, 0, 1, 0, 0, 0, 0], self.n_shape).astype('bool')
            self.y1_mask = np.tile([0, 1, 0, 0, 0, 0, 0], self.n_shape).astype('bool')

            # The population_b10 positions that will become the coordinates (x, y) for the shape centres
            #                     c1,c2,r1, r, g, b, a
            self.c_mask = np.tile([1, 1, 0, 0, 0, 0, 0, 0], self.n_shape).astype('bool')
            # The population_b10 positions that will become the radii in the x, y directions
            self.r_mask = np.tile([0, 0, 1, 1, 0, 0, 0, 0], self.n_shape).astype('bool')

        elif self.shape_type in {'ellipse', 'rectangle'}:
            # The population_b10 positions that will become the coordinates (x, y) for the shape centres
            #                     c1,c2,r1,r2, r, g, b, a
            self.c_mask = np.tile([1, 1, 0, 0, 0, 0, 0, 0], self.n_shape).astype('bool')
            # The population_b10 positions that will become the radii in the x, y directions
            self.r_mask = np.tile([0, 0, 1, 1, 0, 0, 0, 0], self.n_shape).astype('bool')
            # Both of the above
            self.cr_mask = self.c_mask | self.r_mask

        if self.shape_type in {'circle', 'square'}:
            self.rgb_mask = np.tile([0, 0, 0, 0, 1, 1, 1, 0], self.n_shape).astype('bool') # RGB values
            self.a_mask   = np.tile([0, 0, 0, 0, 0, 0, 0, 1], self.n_shape).astype('bool') # Alpha values
        else:
            self.rgb_mask = np.tile(np.hstack([np.zeros(self.n_xy), [1, 1, 1, 0]]), self.n_shape).astype('bool') # RGB values
            self.a_mask   = np.tile(np.hstack([np.zeros(self.n_xy), [0, 0, 0, 1]]), self.n_shape).astype('bool') # Alpha values

    def initialize_svg_template(self):
        '''
        Creates two SVG templates (of type str) in memory. The first template contains the necessary header required
        in a valid SVG image. The second is for creating a shape and depends on the chosen shape type. A copy of template
        2 is added to template 1 for each shape in an individual, creating a master template. Everytime a new shape is 
        generated, a copy of template 2 is added to the master template.
        '''
        if self.shape_type in {'circle', 'ellipse'}:
            self.shape_template = '<ellipse cx="%f" cy="%f" rx="%f" ry="%f" fill="#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n'

        elif self.shape_type in {'square', 'rectangle'}:
            self.shape_template = '<rect x="%f" y="%f" width="%f" height="%f" fill="#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n'

        elif self.shape_type == 'polygon':
            self.shape_template = ('<polygon points="%s" fill="@' % ''.join('%f,' for _ in range(
                self.n_xy))).replace('@', '#{:02x}{:02x}{:02x}" fill-opacity="%f" />\n')

        self.svg_template = '<svg viewBox="0 0 %s %s" xmlns="http://www.w3.org/2000/svg">\n' % self.img_size \
                            + self.svg_bg \
                            + ''.join(self.shape_template for _ in range(self.n_shape))

    def initialize_population(self):
        '''
        Initialize a population at random positions in the search space
        '''
        # Binary representations
        self.population_b2 = np.random.randint(2, size=(self.n_pop, self.n_bits))

        # population_b2 is a 3D array which will contain the population, personal best and global best
        # in binary form. We first take 3 copies of population_b2 to create the correct shape and then
        # replace as we go
        self.population_b2 = np.array([self.population_b2, self.population_b2, self.population_b2], dtype='bool')

        # Initialize p and g distance as inf
        self.g_dist      = np.inf      # global best
        self.prev_g_dist = self.g_dist # Keep track of previous g_best incase a rollback is required
        self.p_dist      = np.full(self.n_pop, np.inf) # personal best

    def crossover(self):
        '''
        Crossover is performed by some indexing of the 3D binary array
        '''
        self.population_b2[0] = self.population_b2[
                                           np.random.choice(3, size=(self.n_pop, self.n_bits), p=self.w),
                                           self.static_idx[0],
                                           self.static_idx[1]
                                           ]

    def crossover_oaat_mode(self):
        '''
        Crossover is performed by some indexing of the 3D binary array
        '''
        self.population_b2[0, :, -self.s_bits: ] = self.population_b2[:, :, -self.s_bits: ][
                                                   np.random.choice(3, size=(self.n_pop, self.s_bits), p=self.w),
                                                   self.static_idx[0],
                                                   self.static_idx[1]
                                                   ]

    def mutation(self):
        self.flip_bits()
        if self.n_shape > 1 and np.random.rand() < self.m[1]:
            self.swap_indices()

    def mutation_oaat_mode(self):
        self.flip_bits_oaat_mode()
        if self.n_shape > 1 and np.random.rand() < self.m[1]:
            self.swap_indices()

    def probabilistic_shape_management(self):
        return np.random.rand() < self.shape_management_prob

    def velocity_shape_management(self):
        return self.iter_with_no_improvement >= self.shape_management_delta

    def periodic_shape_management(self):
        return self.iter % self.shape_management_interval == 0

    def _shape_management(self):
        if self.rollback_mode and self.g_dist >= self.prev_g_dist:
            self.reinitialize_shape()
        else:
            self.initialize_shape()

    def initialize_shape(self):
        '''
        Add a new shape to each individual
        '''
        # Backup current bests
        self.prev_g_dist        = self.g_dist
        self.population_b2[0]   = self.population_b2[2].copy()
        self.population_b2[1]   = self.population_b2[2].copy()
        self.prev_population_b2 = self.population_b2.copy()
        self.n_shape       += 1
        self.svg_template  += self.shape_template
        self.initialize_mapping()
        self.generate_shape()
        self.new_shape_flag = True
        self.n_shape_signal.emit(self.n_shape)

    def reinitialize_shape(self):
        '''
        Reinitialize the previously added shape. This may be required if the algorithm has gotten stuck
        '''
        self.population_b2 = self.prev_population_b2
        self.g_dist        = self.prev_g_dist
        self.initialize_mapping()
        self.generate_shape()
        self.new_shape_flag    = True
        self.reinit_shape_flag = True

    def generate_shape(self):
        '''
        Adds a new shape to population_b2
        '''
        self.population_b2 = np.dstack([self.population_b2, np.random.randint(2, size=(3, self.n_pop, self.s_bits), dtype='bool')])

    def flip_bits(self):
        '''
        Performs some mutation on the population by flipping 1s to 0s,
        or 0s to 1s
        '''
        mutants = np.random.rand(self.n_pop, self.n_bits) < self.m[0]
        self.population_b2[0, mutants] = ~self.population_b2[0, mutants]

    def flip_bits_oaat_mode(self):
        '''
        One At A Time version of self.flip_bits()
        '''
        mutants = np.random.rand(self.n_pop, self.s_bits) < self.m[0]
        self.population_b2[0, :, -self.s_bits : ][mutants] = ~self.population_b2[0, :, -self.s_bits : ][mutants]

    def initialize_individual(self):
        self.w_idx = np.argmax(self.distance)
        self.population_b2[0][self.w_idx] = np.random.randint(2, size=(self.n_bits))

    def swap_indices(self):
        '''
        Swaps some of the parts of the encoded solutions that correspond to shapes
        '''
        mutant = np.random.randint(self.n_pop)
        self.population_b2[0, mutant, :] = self.population_b2[0, mutant, np.random.permutation(self.swap_idx).ravel()]

    def evaluate(self):
        '''
        Measure distances of solutions from reference, rank solutions and update bests
        '''
        self.map_b2_to_b10()
        self.map_to_space()

        pixel_data     = np.array([self.render_img(i) for i in self.pop_range], dtype='int64')
        reference      = self.reference
        distance       = numexpr.evaluate('sum((pixel_data - reference) ** 2.0, 1)', order='C')
        self.distance  = numexpr.evaluate('sum(distance, 1)', order='C')
        # Bool array of personal best locations
        p_idx = self.distance < self.p_dist
        # Find all indices equal to the best and select one at random
        self.g_idx = np.random.choice(
            np.ravel(np.where(self.distance == self.distance[np.argmin(self.distance)])))

        if self.distance[self.g_idx] < self.g_dist or self.new_shape_flag:
            # The global best is copied so that it is the same shape as the population array

            self.population_b2[2]  = (self.population_b2[0][self.g_idx] * np.ones((self.n_pop, self.n_bits), dtype='bool'))
            self.g_dist        = self.distance[self.g_idx]
            # record new best achieved
            self.g_best         = pixel_data[self.g_idx]
            self.g_flag         = True
            self.new_shape_flag = False
            # self.update_distance_indicator.emit(g_dist)
            self.fitness_indicator_signal.emit(np.sqrt(self.g_dist))
            self.iter_with_no_improvement = 0
        else:
            self.iter_with_no_improvement += 1

        # Update personal bests
        self.population_b2[1][p_idx] = self.population_b2[0][p_idx]
        self.p_dist[p_idx] = self.distance[p_idx]

        # Output performance metrics for plotting purposes
        self.output_performance_metrics()

    def evaluate_oaat_mode(self):
        '''
        One At A Time version
        Measure distances of solutions from reference, rank solutions and update bests
        '''

        if self.new_shape_flag or self.bgs is None:
            if self.population_b10 is None or self.reinit_shape_flag:
                self.map_b2_to_b10()
                self.reinit_shape_flag = False
            else:
                self.map_b2_to_b10_oaat_init_mode()
            self.map_to_space()
            # We back these up to use as backgrounds so that we don't have to render them again
            self.bgs  = [self.render_img_oaat_init_mode(i) for i in self.pop_range]
            # We will use these to measure distance from the reference image
            pixel_data = np.array([self.render_img_oaat_exis_mode(i) for i in self.pop_range], dtype='int64')
        else:
            self.map_b2_to_b10_oaat_exis_mode()
            self.map_to_space()
            # Created using the backed up images
            pixel_data = np.array([self.render_img_oaat_exis_mode(i) for i in self.pop_range], dtype='int64')

        reference      = self.reference
        distance       = numexpr.evaluate('sum((pixel_data - reference) ** 2.0, 1)', order='C')
        self.distance  = numexpr.evaluate('sum(distance, 1)', order='C')

        # Bool array of personal best locations
        p_idx = self.distance < self.p_dist
        # Find all indices equal to the best and select one at random
        self.g_idx = np.argmin(self.distance) # np.random.choice(np.ravel(np.where(self.distance==self.distance[np.argmin(self.distance)])))
        # If potential new best is better then current best
        if self.distance[self.g_idx] < self.g_dist or self.new_shape_flag:
            # Replace current best with new best
            self.g_dist         = self.distance[self.g_idx]
            # The global best is copied so that it is the same shape as the population array
            self.population_b2[2]   = (self.population_b2[0][self.g_idx] * np.ones((self.n_pop, self.n_bits), dtype='bool'))
            # record new best achieved
            self.g_flag         = True
            self.new_shape_flag = False
            self.fitness_indicator_signal.emit(np.sqrt(self.g_dist))
            self.iter_with_no_improvement = 0
        else:
            self.iter_with_no_improvement += 1

        # Update personal bests
        self.population_b2[1][p_idx] = self.population_b2[0][p_idx]
        self.p_dist[p_idx]       = self.distance[p_idx]

        # Output performance metrics for plotting purposes
        self.output_performance_metrics()

    def output_performance_metrics(self):
        '''
        For updating a display which monitors progress 
        '''
        distance = np.sqrt(self.distance)
        best     = distance[self.g_idx]
        avg      = np.mean(distance)
        std      = np.std(distance)
        std_hi   = avg + std
        std_lo   = avg - std
        worst    = distance[np.argmax(distance)]
        self.performance_metrics_signal.emit([self.iter, [worst], [std_hi, std_lo], [avg], [best]])

    def b2_to_b10(self, x):
        '''
        For converting an array of binary into an array of base 10 integers. Each time the method
        is called on a column of shape=(n_pop, z_bits) where 'z_bits' can refer to x_bits, y., c., r. etc... 
        Returns a column of single data points shape=(n_pop, 1). To convert entire population_b2 to base 10
        method is called n_shape * n_xyc times.
        '''
        o = np.zeros(x.shape[1], dtype='int64')
        for bits in x:
            o = (o << 1) | bits
        return o

    def map_b2_to_b10(self):
        '''
        Converts the binary array into an array of vectors

        idx - is a list of indices that index the bits of a single base 10 value
        '''
        self.population_b10 = np.array([self.b2_to_b10(self.population_b2[0, :, idx]) for idx in self.b2_idx]).T

    def map_b2_to_b10_oaat_init_mode(self):
        '''
        One At A Time version of map_b2_to_b10()
        Updates population_b10 with the newly INITialized shape rather than recalculating all previous shapes

        idx - is a list of indices that index the bits of a single base 10 value
        '''
        self.population_b10 = np.hstack([self.population_b10, np.array([self.b2_to_b10(self.population_b2[0, :, idx]) for idx in self.b2_idx[-self.n_xyc: ]]).T])

    def map_b2_to_b10_oaat_exis_mode(self):
        '''
        One At A Time version of map_b2_to_b10()
        Updates population_b10 with the changes that have occured in the most recently initialized but EXISting shape

        idx - is a list of indices that index the bits of a single base 10 value
        '''
        self.population_b10[:, self.b10_idx[-1]] = np.array([self.b2_to_b10(self.population_b2[0, :, idx]) for idx in self.b2_idx[-self.n_xyc : ]]).T

    def map_to_space(self):
        self.mapped_population_b10 = (self.population_b10.astype('float64') * self.map).round(0).astype('int64')

    def translate_circles(self, x):
        # Calculate which x2 need clipping
        x[2::7] = (x[2::7] - x[::7]) / 2.
        # Duplicate x2 -> r1, r2
        o = np.empty(x.shape[0] + self.n_shape, dtype='float64')
        o[~self.uni_vert_mask] = x
        o[self.uni_vert_mask] = x[2::7]
        # Centres
        o[self.c_mask] += o[self.r_mask]
        o[self.r_mask] = np.abs(o[self.r_mask])
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
        # Calculate diameter / width
        x[2::7] = x[2::7] - x[::7]
        # Duplicate x2 -> r1, r2
        o = np.empty(x.shape[0] + self.n_shape, dtype='float64')
        o[~self.uni_vert_mask] = x
        o[self.uni_vert_mask] = x[2::7]
        return o

    def translate_ellipses(self, x):
        '''
        The SVG standard requires an ellipse to be represented by its centre coordinates and radii; which
        differs from that of aggdraw where they are represented by the bottom-left and upper-right coordinates
        of its rectangular bounding box

        Input:
            x - array-like, shape=(1, -1), individual solution encoded as a vector

        Output:
            x - array-like, shape=(1, n_shape * (n_vert + 4)), example: [cx1, cy2, rx1, ry2, r, g, b, a,...]
                where cxi, cyi = centre and rxi, ryi = radii
        '''
        radii = (x[self.r_mask] - x[self.c_mask]) / 2.
        x[self.c_mask] += radii        # Ellipse centres
        x[self.r_mask] = np.abs(radii) # Ellipse radii
        return x

    def translate_rectangles(self, x):
        '''
        The SVG standard requires a rectangle to be represented by the x, y coordinates of its upper-left
        corner and its width and height

        Input:
            x - array-like, shape=(1, -1), individual solution encoded as a vector

        Output:
            x - array-like, shape=(1, n_shape * (n_vert + 4)), example: [cx1, cy2, wx1, hy2, r, g, b, a,...]
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
        '''
        Generates an SVG in the form of a single string
        '''
        # SVG standard (non-CSS) expects alpha to be float (0 <= x <= 1)
        g_best = self.mapped_population_b10[self.g_idx].astype('float64')

        if self.translation_method:
            g_best = self.translation_method(g_best)

        g_best[self.a_mask] /= 255.
        svg = self.svg_template.format(*g_best[self.rgb_mask].astype('uint8'))
        svg = svg % tuple(g_best[~self.rgb_mask])
        svg += '</svg>'
        self.svg = svg

    def output_svg(self):
        '''
        Save an SVG of global best solution
        '''
        with open(os.path.join(self.output_dir, str(self.iter) + '.svg'), 'w') as open_file:
            open_file.write(self.svg)

    def render_uni_verts(self, draw_method, shape):
        '''
        For rendering shapes that have a single vertex and a radius such as circles and squares
        '''
        draw_method((*shape[: self.n_xy], shape[1] + shape[2] - shape[0]), Brush(tuple(shape[self.n_xy: -1]), shape[-1]))

    def render_multi_verts(self, draw_method, shape):
        '''
        For rendering shapes that have multiple vertices: rectangles and polygons
        '''
        draw_method(shape[: self.n_xy], Brush(tuple(shape[self.n_xy: -1]), shape[-1]))

    def render_img(self, i: int):
        '''
        Converts an individual into an image
        '''
        self.draws[i][0].rectangle(self.bg_coords, self.bg_brush)
        for idx in self.b10_idx:
            self.render_method(self.draws[i][1], self.mapped_population_b10[i, idx])
        return np.frombuffer(self.draws[i][0].tobytes(), dtype='uint8').reshape(-1, 4, order='C')[:, :3]

    def render_img_oaat_init_mode(self, i: int):
        '''
        Renders all shapes into an image except for the most recently initialized. These images are then used as
        the backgrounds while we optimize the last shape.
        '''
        self.draws[i][0].rectangle(self.bg_coords, self.bg_brush)
        for idx in self.b10_idx[: -1]:
            self.render_method(self.draws[i][1], self.mapped_population_b10[i, idx])
        img = np.frombuffer(self.draws[i][0].tobytes(), dtype='uint8').reshape(self.img_size[1], self.img_size[0], 4, order='C')
        img = Image.fromarray(img, mode='RGBA')
        return img

    def render_img_oaat_exis_mode(self, i: int):
        '''
        One At A Time
        Renders the last image that was initialized
        '''
        img         = self.bgs[i]
        draw        = Draw(img)
        draw.setantialias(False)
        draw_method = getattr(draw, self.draw_method)
        self.render_method(draw_method, self.mapped_population_b10[i, self.b10_idx[-1]])
        return np.frombuffer(draw.tobytes(), dtype='uint8').reshape(-1, 4, order='C')[:,: 3]