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
USE STD.TEXTIO.ALL;
USE IEEE.STD_LOGIC_TEXTIO.ALL;

ENTITY fulltopHDC IS
    GENERIC (
        inbit                  : INTEGER := 8;   -- Number of bits for input
        dimension              : INTEGER := 1000; -- HDC Dimension size
        pruning                : INTEGER := 336; -- Number of efficient dimensions
        logfeature             : INTEGER := 10; -- LOG2(featureSize)
        classes                : INTEGER := 10; -- Number of classes
        featureSize            : INTEGER := 784; -- Number of elements in each input data
        classMemSize           : INTEGER := 7;  -- Length of each segment of classHyper memories
        confCompNum            : INTEGER := 3;  -- Number of confComp modules in the comparator
        rsaZeropadding         : INTEGER := 1;  -- Zero paddings for the sequential adder (RSA)
        comparatorZeroPadding  : INTEGER := 6;  -- Zero paddings for multiplexer in comparators
        logClasses             : INTEGER := 4;  -- ceiling[LOG2(classes)]
        logn                   : INTEGER := 2;  -- ceilingLOG2(#popCounters OR adI)
        IDreminder             : INTEGER := 232; -- Remainder value for ID-level
        IDcoefficient          : INTEGER := 3;  -- Coefficient of ID-level
        lenTKEEP_M             : INTEGER := 1;
        lenTDATA_S             : INTEGER := 8;
        lenTKEEP_S             : INTEGER := 1
    );
    PORT (
        clk        : IN  STD_LOGIC; 
        rst        : IN  STD_LOGIC; 
        TVALID_M   : IN  STD_LOGIC;         
        TDATA_M    : IN  STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
        TKEEP_M    : IN  STD_LOGIC_VECTOR(lenTKEEP_M-1 DOWNTO 0);
        TREADY_S   : IN  STD_LOGIC;   
        TLAST_M    : IN  STD_LOGIC;    
        TREADY_M   : OUT STD_LOGIC;  -- Should always be '1' for DMA only
        TVALID_S   : OUT STD_LOGIC;         
        TLAST_S    : OUT STD_LOGIC;         
        TDATA_S    : OUT STD_LOGIC_VECTOR(lenTDATA_S-1 DOWNTO 0);
        TKEEP_S    : OUT STD_LOGIC_VECTOR(lenTKEEP_S-1 DOWNTO 0)
    );
END ENTITY fulltopHDC;

ARCHITECTURE behavioral OF fulltopHDC IS

    COMPONENT OTFGEn IS
        GENERIC (
            inbit     : INTEGER := 10; 
            d         : INTEGER := 2000; -- Dimension size
            lgf       : INTEGER := 10; -- LOG2(#feature)
            c         : INTEGER := 10; -- Number of classes
            featureSize : INTEGER := 784;
            n         : INTEGER := 9;  -- Memory pointer bit-widths
            adI       : INTEGER := 2;  -- Number of confComp modules
            adz       : INTEGER := 0;  -- Zero padding for RSA
            zComp     : INTEGER := 6;  -- Zero padding for MUX Comparator
            lgCn      : INTEGER := 4;  -- CeilingLOG2(#Classes)
            logn      : INTEGER := 1;  -- MuxCell RSA
            r         : INTEGER := 2;  -- Remainder for ID-level
            x         : INTEGER := 1   -- Coefficient of IDLEVEL
        );
        PORT (
            clk       : IN  STD_LOGIC; 
            rstl      : IN  STD_LOGIC; 
            run       : IN  STD_LOGIC;
            pixel     : IN  STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
            done      : OUT STD_LOGIC;
            TLAST_S   : OUT STD_LOGIC;
            TVALID_S  : OUT STD_LOGIC;
            ready_M   : OUT STD_LOGIC;
            classIndex : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT regOne IS
        GENERIC (
            init : STD_LOGIC := '1'  -- Initial value
        );
        PORT (
            clk       : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din       : IN  STD_LOGIC;
            dout      : OUT STD_LOGIC
        );
    END COMPONENT;
        
    SIGNAL pixelIn     : STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);
    SIGNAL classIndex  : STD_LOGIC_VECTOR(logClasses - 1 DOWNTO 0);
    SIGNAL rstl, run, done : STD_LOGIC;
    SIGNAL outreg0     : STD_LOGIC_VECTOR(31 DOWNTO 0) := (others => '0');
    SIGNAL pixelreg    : STD_LOGIC_VECTOR(inbit-1 DOWNTO 0);

    CONSTANT ALLZERO : STD_LOGIC_VECTOR(lenTDATA_S-logClasses-1 DOWNTO 0) := (others => '0');

    TYPE state IS (init, registering);
    SIGNAL ns, ps : state;

BEGIN

    -- Instantiating HDC OTFGEn
    HDCOTFGEn : OTFGEn 
        GENERIC MAP (
            inbit      => inbit, 
            d          => dimension, 
            lgf        => logfeature, 
            c          => classes, 
            featureSize => featureSize, 
            n          => classMemSize, 
            adI        => confCompNum, 
            adz        => rsaZeropadding, 
            zComp      => comparatorZeroPadding, 
            lgCn       => logClasses, 
            logn       => logn, 
            r          => IDreminder, 
            x          => IDcoefficient
        )
        PORT MAP (
            clk       => clk, 
            rstl      => rst, 
            run       => run, 
            pixel     => pixelIn, 
            done      => done,  
            TLAST_S   => TLAST_S, 
            TVALID_S  => TVALID_S, 
            ready_M   => TREADY_M,  
            classIndex => classIndex
        );

    -- Assignments
    pixelIn <= TDATA_M;
    run     <= TVALID_M; 
    TDATA_S <= ALLZERO & classIndex;
    TKEEP_S <= (others => '1');

    -- State Machine
    PROCESS(clk) 
    BEGIN 
        IF rising_edge(clk) THEN
            IF (rst = '1') THEN
                ps <= init; 
            ELSE  
                ps <= ns;  
            END IF;
        END IF;
    END PROCESS;

END ARCHITECTURE behavioral;
