-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

use STD.textio.all;
use ieee.std_logic_textio.all;

ENTITY fulltopHDC IS
    GENERIC 
    (	 inbit		:INTEGER  := 8; -- Defines the number of bits for input
        dimension           : INTEGER := 1000; -- The HDC Dimension size
		pruning		: INTEGER := 336; -- The number of efficient dimensions
        logfeature         : INTEGER := 10; -- LOG2(featureSize)
        classes           : INTEGER := 10; ---- Number of classes
        featureSize : INTEGER := 784; --The number of elements in each input data
        classMemSize           : INTEGER := 7; -- The length of each segment of classHyper memories
        confCompNum         : INTEGER := 3; -- Number of confComp modules in the comparator, calculated as ceiling(dimension/(2^classMemSize))
        rsaZeropadding         : INTEGER := 1; -- The number of zero paddings for the sequential adder (RSA)
        comparatorZeroPadding       : INTEGER := 6; -- The number of zero paddings for the multiplexer in comparators
        logClasses        : INTEGER := 4; -- ceiling[LOG2(classes)]
		logn        : INTEGER := 2; -- ceilingLOG2(#popCounters OR adI)
		IDreminder           : INTEGER := 232;                  -- Remainder value for ID-level
	    IDcoefficient           : INTEGER := 3 -- The coefficient of ID-level
		lenTKEEP_M			:INTEGER := 1;
		lenTDATA_S			:INTEGER := 8;
		lenTKEEP_S			:INTEGER := 1;
	);
    PORT
    (
        clk		    : IN STD_LOGIC; 
        rst		    : IN STD_LOGIC; 
        TVALID_M         : IN STD_LOGIC;         
        TDATA_M		: IN STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
        TKEEP_M		: IN STD_LOGIC_VECTOR(lenTKEEP_M-1 DOWNTO 0);
        TREADY_S        : IN STD_LOGIC;   
        TLAST_M        : IN STD_LOGIC;    
        TREADY_M        : OUT STD_LOGIC;    -- should be always '1' as of now! for DMA only
        TVALID_S         : OUT STD_LOGIC;         
        TLAST_S         : OUT STD_LOGIC;         
        TDATA_S  : OUT STD_LOGIC_VECTOR(lenTDATA_S-1 DOWNTO 0);
        TKEEP_S  : OUT STD_LOGIC_VECTOR(lenTKEEP_S-1 DOWNTO 0)
    );
END ENTITY fulltopHDC;

ARCHITECTURE behavioral OF fulltopHDC IS

component OTFGEn IS
    GENERIC
    (	 inbit		:INTEGER  := 8; -- consider 8 bit is enough for grayscale --- it is not
        d           : INTEGER := 1000; -- dimension size
		sparse		: INTEGER := 336; -- after sparse
        lgf         : INTEGER := 10; -- bit width out popCounters --- LOG2(#feature)
        c           : INTEGER := 10; ---- #Classes
        featureSize : INTEGER := 784;
        n           : INTEGER := 7; --512 each classMem -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,, for comparitor thinpg! 256 unit in each portin of memory
        adI         : INTEGER := 3; -- number of confComp module, or adderInput and = ceiling(D/(2^n))
        adz         : INTEGER := 48; -- zeropadding for RSA = 2**? - adI  ====! EXTRA!!! 
        zComp       : INTEGER := 6; -- zeropadding Mux Comp = 2**? - c
        lgCn        : INTEGER := 4; -- ceilingLOG2(#Classes)
		logn        : INTEGER := 2; -- MuxCell RSA, ceilingLOG2(#popCounters OR adI)
		r           : INTEGER := 232;                  -- remainder from division for ID level
	    x           : INTEGER := 3 -- coefficient of IDLEVEL
	);
    PORT
    (
         clk		    : IN STD_LOGIC; 
        rstl		    : IN STD_LOGIC; 
        run         : IN STD_LOGIC;
        pixel		: IN STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
        --update		: IN STD_LOGIC;		
        done        : OUT STD_LOGIC;
        TLAST_S, TVALID_S, ready_M       : OUT STD_LOGIC;
        --pixelMemOutIndex : OUT STD_LOGIC_VECTOR(14 DOWNTO 0);
        classIndex  : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0)
    );
END component;

COMPONENT regOne IS
	GENERIC (init : STD_LOGIC := '1');   -- initial value
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC;
		dout        : OUT  STD_LOGIC
	);
END COMPONENT;
        
signal pixelIn		: STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
signal classIndex  : STD_LOGIC_VECTOR(logClasses - 1 DOWNTO 0);

signal rstl, run, done : STD_LOGIC;
signal outreg0 : std_logic_vector (31 DOWNTO 0):= (others =>'0');
signal pixelreg		: STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
  

CONSTANT ALLZERO := STD_LOGIC_VECTOR(lenTDATA_S-logClasses-1 DOWNTO 0);

TYPE state IS  (init,  registering);
SIGNAL ns,  ps : state;
-- attribute MARK_DEBUG : string;
-- attribute MARK_DEBUG of TVALID_M : signal is "TRUE";
-- attribute MARK_DEBUG of TDATA_M : signal is "TRUE";
-- --attribute MARK_DEBUG of pixelMemOutIndex : signal is "TRUE";
-- attribute MARK_DEBUG of TREADY_S : signal is "TRUE";
-- attribute MARK_DEBUG of TLAST_M : signal is "TRUE";
-- attribute MARK_DEBUG of TREADY_M : signal is "TRUE";
-- attribute MARK_DEBUG of TVALID_S : signal is "TRUE";
-- attribute MARK_DEBUG of TLAST_S : signal is "TRUE";
-- attribute MARK_DEBUG of TDATA_S : signal is "TRUE";
-- attribute MARK_DEBUG of classIndex : signal is "TRUE";
-- attribute MARK_DEBUG of done : signal is "TRUE";
-- attribute MARK_DEBUG of ns : signal is "TRUE";
      
BEGIN

    HDCOTFGEn: OTFGEn 
    GENERIC MAP
    (	 
    inbit, d,pruning, logfeature, classes, featureSize, classMemSize, confCompNum, rsaZeropadding, comparatorZeroPadding, logClasses, logn, IDreminder, IDcoefficient 
	)
    PORT MAP
    (
        clk, rst, run, 
        pixelIn, done,   TLAST_S, TVALID_S, TREADY_M,  
        classIndex 
    );
     
    pixelIn <= TDATA_M;
	run <= TVALID_M; 
 
    TDATA_S <= ALLZERO & classIndex;
    TKEEP_S <= (Others => '1');
    
    PROCESS(clk) BEGIN 
		IF rising_edge(clk) then
			IF (rst ='1')then
				ps <= init; 
			ELSE  
				ps <= ns;  
			END IF;
		END IF;
	END PROCESS;


End architecture;