#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import threading
import time
import serial
import struct

class PythonSerialDriver():
    def __init__(self):
        self.windows = False
        # (38400Kbit/s, 8,N,1)
        baud = 38400
        baseports = ['/dev/ttyUSB', '/dev/ttyACM', 'COM', '/dev/tty.usbmodem1234','/dev/ttyS']
        self.ser = None
        
        while not self.ser:
            for baseport in baseports:
                if self.ser:
                    break
                for i in range(0, 64):
                    try:
                        port = baseport + str(i)
                        self.ser = serial.Serial(port, baud, timeout=1) # 1s timeout
                        print repr('Driver: Opened ' + port + '\r')
                        break
                    except:
                        self.ser = None
                        pass
        
            if not self.ser:
                    print repr('Driver: Could not open a serial port!!!')
                    sys.exit(1)      

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()               
        
        self.COM_RLF = b'\xfc\xff\xff\xff' # pressure relief
        self.COM_P = b'\xfa\x00\xff\xff' # positvie pressure, default gear stratup
        self.COM_N = b'\xfb\x00\xff\xff' # negtive pressure, default gear stratup
        self.COM_RD_P = b'\xaa\xff\xff\x16' # read PPI
        self.COM_W_P = bytearray(b'\xea\x00\x32\x16') # write PPI, default as 50
        self.COM_RD_N = b'\xab\xff\xff\x16' # read NPI
        self.COM_W_N = bytearray(b'\xeb\x11\x3c\x16') # write NPI, default as -60 
        self.COM_RD = b'\xa9\xff\xff\x16' # read real-time pressure 
        
        # TODO: class SRTComm(){}
        self.records = b'\xff\xff\xff\xff'
        self.commands = b'\xff\xff\xff\xff'
        
        self.RE_LEN = 4     # N.B. TWC and PWC reply byte == 6        
        # TODO: enum {pos, neg, rlf} flag
        self.FLG_POS = 1
        self.FLG_NEG = -1
        self.FLG_ZERO = 0
        self.flag = self.FLG_ZERO

        self.DELAY = 0.6    # s        
        self.TIMEOUT = 2
        self.OFFSET = 5    # kPa
        self.PPI = 60
        self.NPI = -60        
        # self.PI = self.NPI

        self.hexstring = b'\xfc\xff\xff\xff'
        # TODO: enum {r_ppi, w_ppi, ...} func
        self.func = 0xfc            
        # TODO: enum {pos, neg} sign
        self.sign = 0x00        # 0x11 for negative, 0x00 for positive
        # self.pressure = '%02x' % (self.PPI)    # [0, 95]
        self.pressure = 0    # int in decimal format, [0, 95]
        # TODO: enum {hex, dec} base
        self.base = 0x16

        self.counter = 10   # loopTest       
        self.printLog = True   

    def send(self, data):
        self.commands = data
        try:                                
            self.ser.write(self.commands)            
            self.ser.flush()                                                         
            if self.printLog:
                if self.commands == self.COM_RD:
                    print repr('Output: inquire about RTP')                      
                else:
                    print repr('Output: ' + str(self.commands))                      
        except IOError:
            # Manually raise the error again so it can be caught outside of this method
            # raise IOError()               
            print ('Driver: Serial port disconnected')   # handle IO err here if any
            return -1

    def receive(self):             
        t0 = time.time()
        while True:
            if time.time() - t0 >= self.TIMEOUT:                
                return -1          
            len = self.ser.inWaiting()
            if len >= 4:                                                 
                if self.commands == self.COM_W_N or self.commands == self.COM_RD_N or self.commands == self.COM_W_P or self.commands == self.COM_RD_P or self.commands == self.COM_RD:
                    #self.func, self.sign, self.pressure, self.base = struct.unpack('!BBBB', self.ser.read(self.RE_LEN))
                    #self.func, self.sign, self.pressure, self.base  = self.ser.read(self.RE_LEN).strip('b').split('\'
                    #self.func, self.sign, self.pressure, self.base = bytearray.fromhex( self.ser.read)
                    # print repr('-------------------func[' + str(self.func) + ']-------------------' )
                    # self.func, self.sign, self.pressure, self.base = self.ser.read(self.RE_LEN)
                    # # [self.hexstring[i:i+4] for i in range(0,self.RE_LEN, 2)]
                    # # self.func = self.hexstring[0]
                    # # self.sign = self.hexstring[1]
                    # # self.pressure = self.hexstring[2]
                    # # self.base = self.hexstring[3]
                    # tmp = self.commands
                    # if self.func != tmp[0].fromhex():                    
                    #     print repr('Func codes must be identical ' + str(self.func))
                    #     print repr('func[' + str(self.func) + '] vs com[' +  tmp[0] + ']' )                        
                    #     return -1
                    
                    if self.sign != 0x00 and self.sign != 0x11:
                        print repr('Sign value can only be 0x00 or 0x11')
                        return -1
                    elif self.func == 0xee or self.sign == 0xee or self.pressure == 0xee or self.base == 0xee:
                        print repr('Check commands format or serial cable')                
                        return -1
                    else:
                        if self.printLog:
                            print repr('Input: RTP = %s%d' % ('-' if self.sign == 0x11 else '', self.pressure))    
                else: # TODO: add other comm validation?
                    self.records = self.ser.read(self.RE_LEN)                  
                    if self.printLog:
                        print repr('Input: ' + str(self.records))                    
                    
                self.ser.reset_input_buffer()
                return                

    def comm(self, data, delay = 0):                                             
        t0 = time.time()        
        while True:
            if time.time() - t0 >= (2 * self.TIMEOUT):                
                print ('Driver: Communication failed, current command <%s>' %(self.commands))
                return -1
            else:                                
                bErr = (True if self.send(data) == -1 else False)                
                time.sleep(delay)
                if (True if self.receive() == -1 else True if bErr else False): 
                    if self.printLog:
                        print repr('Comm: retry')               
                    continue
                else:                    
                    break              

    def setMove(self, pressure, delay):
        self.DELAY = delay               
        if pressure <= 0:
            if pressure == 0:
                # if self.comm(self.COM_W_N) == -1:
                #     return -1
                return
            elif pressure < -70:
                pressure = -65            
                print ('Driver: NPI only works below -70, reset to -65')                
            self.COM_W_N[2] = int(abs(pressure))
            if self.comm(self.COM_W_N) == -1:
                print repr('Write NPI failed')
                return -1
            self.NPI = -self.pressure            
            if self.comm(self.COM_RD_N) == -1:
                print repr('Read NPI failed')
                return -1            
            if self.printLog:
                print repr('--- NPI is %s ---' % (self.NPI))                                    
        else:            
            if pressure < 5:
                pressure = 5  
                print ('Driver: PPI too low, reset to 5')                              
            elif pressure > 95:
                pressure = 90
                print ('Driver: PPI only works below 95, reset to 90')                
            self.COM_W_P[2] = int(pressure)  
            if self.comm(self.COM_W_P) == -1:
                print repr('Write PPI failed')
                return -1
            self.PPI = self.pressure                        
            if self.comm(self.COM_RD_P) == -1:
                print repr('Read PPI failed')
                return -1            
            if self.printLog:
                print repr('--- PPI is %s ---' % (self.PPI))                
    
    def isReady(self):        
        if self.comm(self.COM_RD) == -1:
            print repr('Read RTP failed')
            return -1
        reading = self.pressure        
        if self.flag == self.FLG_ZERO:
            if abs(reading) > self.OFFSET:
                return -1
            return                                                                    
        elif self.flag == self.FLG_NEG:
            if abs(reading + self.NPI) > self.OFFSET:
                # self.PI = self.NPI
                # self.setMove(self.NPI + (reading + self.NPI)/2)                
                return -1
            return                    
        elif self.flag == self.FLG_POS:
            if abs(reading - self.PPI) > self.OFFSET:
                # self.PI = self.PPI
                # self.setMove(self.PPI - (reading - self.PPI)/2)                
                return -1
            return            
        else:
            # only check incorrect flag format here
            raise ValueError('isReady: Wrong flag: %d' %(self.flag))                        
           
    def move(self, flag, bRtFeedback = True, bRlfReset = True):
        self.flag = flag
        bErr = False
        t0 = time.time()        
        while True:
            if time.time() - t0 >= (2 * self.TIMEOUT):                                
                print repr('Move ready timeout, RTP = %s%d' % ('-' if self.sign == 0x11 else '', self.pressure))
                return -1
            else:                 
                if self.printLog:
                    print repr('<<<[MOVE TO %s = %d]>>>' % ('PPI' if self.flag == self.FLG_POS else 'NPI' if self.flag == self.FLG_NEG else 'ZERO', self.PPI if self.flag == self.FLG_POS else self.NPI if self.flag == self.FLG_NEG else 0))                
                if (bRlfReset and bErr) or self.flag == self.FLG_ZERO:
                    self.flag = self.FLG_ZERO                
                    if self.comm(self.COM_RLF, self.DELAY) == -1 or (bRtFeedback and self.isReady() == -1):                
                        print repr('Move: relieve failed, retry')
                        continue
                    else:
                        self.flag = flag
                        bErr = False
                if self.flag == self.FLG_NEG:                
                    if self.comm(self.COM_N, self.DELAY) == -1 or (bRtFeedback and self.isReady() == -1):                
                        bErr = True
                        print repr('Move: move to NPI failed, retry')            
                        continue                    
                elif self.flag == self.FLG_POS:                          
                    if self.comm(self.COM_P, self.DELAY) == -1 or (bRtFeedback and self.isReady() == -1):
                        bErr = True
                        print repr('Move: move to PPI failed, retry')
                        continue
                # self.setMove(self.PI)                
                break        

    def moveTo(self, where = None, pressure = 0, delay = None, bRtFeedback = True, bRlfReset = True):
        if where is None:
            where = self.FLG_ZERO 
        if delay is None:
            delay = self.DELAY             
        if self.setMove(pressure, delay) == -1:
            return -1
        try:
            return self.move(where, bRtFeedback)
        except ValueError:
            raise ValueError()                

    def loopTestRTFRetry(self, bRtFeedback = True, bRlfReset = True):                                   
        cnt = self.counter             
        delay = self.DELAY  
        offset = 20
        step_pos = (self.PPI - offset)/self.counter
        step_neg = (self.NPI + offset)/self.counter        
        t0 = time.time()     
        
        while cnt:   
            if bRtFeedback:         
                self.moveTo(self.FLG_ZERO)
                self.moveTo(self.FLG_NEG, step_neg * (self.counter - cnt) - offset, float(cnt / self.counter * delay / 20 * 6  + delay/2), True, bRlfReset)
                self.moveTo(self.FLG_ZERO)
                self.moveTo(self.FLG_POS, offset + step_pos * (self.counter - cnt), float(cnt / self.counter * delay / 20 * 4  + delay/2), True, bRlfReset)            
            else:
                self.moveTo(self.FLG_ZERO, 0, None, False)
                self.moveTo(self.FLG_NEG, step_neg * (self.counter - cnt) - offset, float(cnt / self.counter * delay / 20 * 6  + delay/2), False)
                self.moveTo(self.FLG_ZERO, 0, None, False)
                self.moveTo(self.FLG_POS, offset + step_pos * (self.counter - cnt), float(cnt / self.counter * delay / 20 * 4  + delay/2), False)
            
            cnt -= 1                        

        print repr('--- %s seconds for 1 cycle ---' % ((time.time() - t0)/(self.counter)))
