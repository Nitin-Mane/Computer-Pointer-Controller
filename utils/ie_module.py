#!/usr/bin/env python3

######################################################################################################
#
#                                     IE Module Script
#
######################################################################################################

'''
OpenVINO IE Module Script: 

This script represents the inference classe of the plugin devices and based on the hardware selection the 
operation will be executed with the help of the OpenVINO libraries.

Device = {CPU, GPU, FPGU, MYRID, HDDL, VPU}

[Doc](https://docs.openvinotoolkit.org/2018_R5/_docs_IE_DG_inference_engine_intro.html)
'''
# Load the library
import logging as log
import os.path as osp

# load OpenVINO library
from openvino.inference_engine import IEPlugin

class Inference_Context:
    '''
    Inference Enginer API 

    Loading the Plugins for the hardware inputs. 
    '''
    def __init__(self):
        
        # Initialize the plugins
        self.plugins = {}

    def load_plugins(self, devices, cpu_ext="", gpu_ext=""):
        
        # Load the plugin device as CPU or GPU instruction
        log.info("Loading plugins for devices: %s" % (devices))

        plugins = { d: IEPlugin(d) for d in devices }
        
        # if the plugin selected then the device fetech the instruction path
        # for CPU Hardware
        if 'CPU' in plugins and not len(cpu_ext) == 0:
            log.info("Using CPU extensions library '%s'" % (cpu_ext))
            assert osp.isfile(cpu_ext), "Failed to open CPU extensions library"
            plugins['CPU'].add_cpu_extension(cpu_ext)
        
        # for GPU Hardware
        if 'GPU' in plugins and not len(gpu_ext) == 0:
            assert osp.isfile(gpu_ext), "Failed to open GPU definitions file"
            plugins['GPU'].set_config({"CONFIG_FILE": gpu_ext})
        
        # set the plugins
        self.plugins = plugins
        
        # generating the log information of the plugins loaded
        log.info("Plugins are loaded")

    def get_plugin(self, device):
        
        # getting the plugin device
        return self.plugins.get(device, None)

    def check_model_support(self, net, device):
        
        '''
        Check the model support
        This is important steps for the device to process the inference API pipeline
        '''
        
        plugin = self.plugins[device]

        
        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() \
                                    if l not in supported_layers]

            if len(not_supported_layers) != 0:
                log.error("The following layers are not supported " \
                    "by the plugin for the specified device {}:\n {}" \
                    .format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions " \
                    "library path in the command line parameters using " \
                    "the '-l' parameter")

                raise NotImplementedError(
                    "Some layers are not supported on the device")

    def load_model(self, model, device, max_requests=1):
        
        # Load the model with the device
        self.check_model_support(model, device)
        plugin = self.plugins[device]
        
        # plugin process for the API request
        deployed_model = plugin.load(network=model, num_requests=max_requests)
        return (deployed_model)

class Module(object):
    '''
    Module Class: 
    This class represent the technical aspects of the modules for processing the data and model. 
    the request which are active and process get the output and performance results. 

    It provide performance and deployment results.
    '''
    def __init__(self, model):
        
        # Initialize the model
        self.model = model
        self.device_model = None
        
        # setting the request to 0 for inital stage
        self.max_requests = 0
        self.active_requests = 0
        
        # clearing the points
        self.clear()

    def deploy(self, device, context, queue_size=1):
        
        # deploy the context and the device with 1 queue process
        self.context = context
        self.max_requests = queue_size
        
        # load the model with the device input instruction
        self.device_model = context.load_model(
            self.model, device, self.max_requests)
        self.model = None

    def enqueue(self, input):
        
        self.clear()
        
        # clear the results
        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False
        
        # device model async operation
        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return (True)

    def wait(self):
        
        # wait list
        if self.active_requests <= 0:
            return ()

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            self.outputs[i] = self.device_model.requests[i].outputs
            self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()
        
        # if the wait request are process finishes
        self.active_requests = 0

    def get_outputs(self):
        
        # output results from the wait request complete cycle for operation
        self.wait()
        return (self.outputs)

    def get_performance_stats(self):
        
        # Getting the performace statiscs results
        return (self.perf_stats)


    def clear(self):
        
        # Clearing the stats and output results
        self.perf_stats = []
        self.outputs = []