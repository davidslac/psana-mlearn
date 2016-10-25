class Vgg16(Pipeline):
    def __init__(self):
        super(Vgg16, self).__init__( **kwargs)

    def get_parser(self, **kwargs):
        parser = super(Vgg16, self).get_parser(**kwargs)
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')
  
    def init(self,args, data_generator, img2vgg16, img2vgg16params=None):
        super(Vgg16Pipeline, self).init(args=args, data_generator=data_generator)
        self.img2vgg16 = img2vgg16
        self.img2vgg16params = img2vgg16params
        self.hdr='Vgg16Pipeline'
        self._vgg = None

    def vgg(self):
        if self._vgg is None:
            self.trace('creating vgg16 from weights=%s' % self.args.vgg16weights)
            self._vgg = psmlearn.vgg16.create(session=self.session, 
                                              weights=self.args.vgg16weights)
        return self._vgg

    def getParser(self, outputdir):
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')

    def data_stats(self, data_iter, input_files, output_files):
        means = []
        for imgNum,img in enumerate(data_iter):
            vgg16img = self.img2vgg16(img=img, channel_mean=None, **self.img2vgg16params)
            means.append(np.mean(vgg16img))
            self.debug('data_stats: img %5d mean=%.2f' % (imgNum, means[-1]))
        channel_mean = np.mean(np.array(means))
        h5util.dict2h5(output_files[0], {'channel_mean':channel_mean,
                                         'number_images':(imgNum+1)})
        self.trace('data_stats: finished - final channel mean=%.2f' % channel_mean)

    def model_layers(self, data_iter, input_files, output_files):
        vgg = self.vgg()
        h5in = h5py.File(input_files[0],'r')
        channel_mean = h5in['channel_mean'].value
        num_images = h5in['number_images'].value
        self.trace('write_model_layers: starting - loaded channel mean=%.2f, num_images=%d' % (channel_mean, num_images))
        h5 = h5py.File(output_files[0],'w')
        first = True
        for img in data_ter:            
            preprocessed_img = self.preprocess_img(img=img, 
                                                   channel_mean=channel_mean, 
                                                   **self.preprocess_params)
#            fc1, fc2 = vgg.get_output_layers(vgg16i
                
            first = False


'''
class Vgg16(Pipeline):
    def __init__(self):
        super(Vgg16, self).__init__( **kwargs)

    def get_parser(self, **kwargs):
        parser = super(Vgg16, self).get_parser(**kwargs)
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')
  
    def init(self,args, data_generator, img2vgg16, img2vgg16params=None):
        super(Vgg16Pipeline, self).init(args=args, data_generator=data_generator)
        self.img2vgg16 = img2vgg16
        self.img2vgg16params = img2vgg16params
        self.hdr='Vgg16Pipeline'
        self._vgg = None

    def vgg(self):
        if self._vgg is None:
            self.trace('creating vgg16 from weights=%s' % self.args.vgg16weights)
            self._vgg = psmlearn.vgg16.create(session=self.session, 
                                              weights=self.args.vgg16weights)
        return self._vgg

    def getParser(self, outputdir):
        parser.add_argument('--vgg16weights', type=str, help='weights file for vgg16', default='data/vgg16_weights.npz')

    def data_stats(self, data_iter, input_files, output_files):
        means = []
        for imgNum,img in enumerate(data_iter):
            vgg16img = self.img2vgg16(img=img, channel_mean=None, **self.img2vgg16params)
            means.append(np.mean(vgg16img))
            self.debug('data_stats: img %5d mean=%.2f' % (imgNum, means[-1]))
        channel_mean = np.mean(np.array(means))
        h5util.dict2h5(output_files[0], {'channel_mean':channel_mean,
                                         'number_images':(imgNum+1)})
        self.trace('data_stats: finished - final channel mean=%.2f' % channel_mean)

    def model_layers(self, data_iter, input_files, output_files):
        vgg = self.vgg()
        h5in = h5py.File(input_files[0],'r')
        channel_mean = h5in['channel_mean'].value
        num_images = h5in['number_images'].value
        self.trace('write_model_layers: starting - loaded channel mean=%.2f, num_images=%d' % (channel_mean, num_images))
        h5 = h5py.File(output_files[0],'w')
        first = True
        for img in data_ter:            
            preprocessed_img = self.preprocess_img(img=img, 
                                                   channel_mean=channel_mean, 
                                                   **self.preprocess_params)
#            fc1, fc2 = vgg.get_output_layers(vgg16i
                
            first = False
'''
