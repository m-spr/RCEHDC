LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

use STD.textio.all;
use ieee.std_logic_textio.all;

ENTITY sparseltopHDC IS
    GENERIC
    (	pixbit		:INTEGER  := 10; -- consider 8 bit is enough for grayscale --- it is not
        d           : INTEGER := 2000; -- dimension size
		sparse		: INTEGER := 666; -- after sparse
        lgf         : INTEGER := 10; -- bit width out popCounters --- LOG2(#feature)
        c           : INTEGER := 10; ---- #Classes
        featureSize : INTEGER := 784;
        n           : INTEGER := 9; --512 each classMem -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,, for comparitor thinpg! 256 unit in each portin of memory
        adI         : INTEGER := 2; -- number of confComp module, or adderInput and = ceiling(D/(2^n))
        adz         : INTEGER := 0; -- zeropadding for RSA = 2**? - adI
        zComp       : INTEGER := 6; -- zeropadding Mux Comp = 2**? - c
        lgCn        : INTEGER := 4; -- ceilingLOG2(#Classes)
		logn        : INTEGER := 1; -- MuxCell RSA, ceilingLOG2(#popCounters OR adI)
		x           : INTEGER := 1 -- coefficient of IDLEVEL
	);
    PORT
    (
        clk		    : IN STD_LOGIC; 
        rst		    : IN STD_LOGIC; 
        TVALID_M         : IN STD_LOGIC;         
        TDATA_M		: IN STD_LOGIC_VECTOR(31 DOWNTO 0);
        TKEEP_M		: IN STD_LOGIC_VECTOR(3 DOWNTO 0);
        TREADY_S        : IN STD_LOGIC;   
        TLAST_M        : IN STD_LOGIC;    
        TREADY_M        : OUT STD_LOGIC;    -- should be always '1' as of now! for DMA only
        TVALID_S         : OUT STD_LOGIC;         
        TLAST_S         : OUT STD_LOGIC;         
        TDATA_S  : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
        TKEEP_S  : OUT STD_LOGIC_VECTOR(3 DOWNTO 0)
    );
END ENTITY sparseltopHDC;

ARCHITECTURE behavioral OF sparseltopHDC IS

component spaeseOTFGEn IS
    GENERIC
    (	pixbit		:INTEGER  := 10; -- consider 8 bit is enough for grayscale --- it is not
        d           : INTEGER := 2000; -- dimension size
		sparse		: INTEGER := 666; -- after sparse
        lgf         : INTEGER := 10; -- bit width out popCounters --- LOG2(#feature)
        c           : INTEGER := 10; ---- #Classes
        featureSize : INTEGER := 784;
        n           : INTEGER := 9; --512 each classMem -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,, for comparitor thinpg! 256 unit in each portin of memory
        adI         : INTEGER := 2; -- number of confComp module, or adderInput and = ceiling(D/(2^n))
        adz         : INTEGER := 0; -- zeropadding for RSA = 2**? - adI
        zComp       : INTEGER := 6; -- zeropadding Mux Comp = 2**? - c
        lgCn        : INTEGER := 4; -- ceilingLOG2(#Classes)
		logn        : INTEGER := 1; -- MuxCell RSA, ceilingLOG2(#popCounters OR adI)
		x           : INTEGER := 1 -- coefficient of IDLEVEL
	);
    PORT
    (
         clk		    : IN STD_LOGIC; 
        rstl		    : IN STD_LOGIC; 
        run         : IN STD_LOGIC;
        pixel		: IN STD_LOGIC_VECTOR(pixbit-1 DOWNTO 0);
        --update		: IN STD_LOGIC;		
        done        : OUT STD_LOGIC;
        --memen       : OUT STD_LOGIC;
        --pixelMemOutIndex : OUT STD_LOGIC_VECTOR(14 DOWNTO 0);
        classIndex  : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0)
    );
END component;
        
signal pixelIn		: STD_LOGIC_VECTOR(pixbit-1 DOWNTO 0);
signal classIndex  : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);

signal rstl, run, done : STD_LOGIC;
signal outreg0 : std_logic_vector (31 DOWNTO 0):= (others =>'0');
signal pixelreg		: STD_LOGIC_VECTOR(pixbit-1 DOWNTO 0);
  

attribute MARK_DEBUG : string;
attribute MARK_DEBUG of TVALID_M : signal is "TRUE";
attribute MARK_DEBUG of TDATA_M : signal is "TRUE";
--attribute MARK_DEBUG of pixelMemOutIndex : signal is "TRUE";
attribute MARK_DEBUG of TREADY_S : signal is "TRUE";
attribute MARK_DEBUG of TLAST_M : signal is "TRUE";
attribute MARK_DEBUG of TREADY_M : signal is "TRUE";
attribute MARK_DEBUG of TVALID_S : signal is "TRUE";
attribute MARK_DEBUG of TLAST_S : signal is "TRUE";
attribute MARK_DEBUG of TDATA_S : signal is "TRUE";
attribute MARK_DEBUG of classIndex : signal is "TRUE";
      
TYPE state IS  (init,  registering);
SIGNAL ns,  ps : state;
BEGIN
--rstl <= not(rst);

    HDCOTFGEn: spaeseOTFGEn 
    GENERIC MAP
    (	 10, 1000, 666, 10, 10, 784, 9, 2, 0, 6, 4, 1, 1
    --pixbit, d, sparse, lgf, c, featureSize, n, adI, adz, zComp, lgCn, logn, x  
	)
    PORT MAP
    (
        clk, rst, run, 
        pixelIn, done,    
        classIndex 
    );
     
    pixelIn <= TDATA_M(pixbit-1 DOWNTO 0);
	run <= TVALID_M; 
    
    --TREADY_M <= not(TLAST_M);
    TREADY_M <= '1';
    
    TDATA_S <= outreg0(31 downto lgCn) & classIndex;
    TKEEP_S<= "0111";
    
    PROCESS(clk) BEGIN 
		IF rising_edge(clk) then
			IF (rst ='1')then
				ps <= init; 
			ELSE  
				ps <= ns;  
			END IF;
		END IF;
	END PROCESS;
	
	PROCESS ( ps,  done)
	BEGIN 
	TLAST_S <= '0';
    TVALID_S <= '0';
		CASE (ps) IS 
			WHEN init =>
            IF ( done = '1') THEN
                    ns <= registering;
            END IF;
            
			WHEN registering =>
                TLAST_S <= '1';
                TVALID_S <= '1';
                IF (TREADY_S = '1') THEN  --- perhaps -1 is extra! check
                    ns <= init;
				ELSE
					ns <= registering;
				END IF;
			WHEN OTHERS =>
					ns <= init;
		END CASE;
	END PROCESS;


End architecture;