LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

use STD.textio.all;
use ieee.std_logic_textio.all;

ENTITY OTFGEn IS
    GENERIC
    (	 pixbit		:INTEGER  := 8; -- consider 8 bit is enough for grayscale --- it is not
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
        pixel		: IN STD_LOGIC_VECTOR(pixbit-1 DOWNTO 0);
        --update		: IN STD_LOGIC;		
        done        : OUT STD_LOGIC;
        TLAST_S, TVALID_S, ready_M       : OUT STD_LOGIC;
        --pixelMemOutIndex : OUT STD_LOGIC_VECTOR(14 DOWNTO 0);
        classIndex  : OUT STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0)
    );
END ENTITY OTFGEn;

ARCHITECTURE behavioral OF OTFGEn IS

signal rst : std_logic;

COMPONENT regOne IS
	GENERIC (init : STD_LOGIC := '1');   -- initial value
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC;
		dout        : OUT  STD_LOGIC
	);
END COMPONENT;
COMPONENT connector IS 
	GENERIC(d : INTEGER := 1000; ----dimentionsize 
	p: INTEGER:= 1000 );--- prunsize 
	PORT ( 
		input         : IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0); 
		pruneoutput        : OUT  STD_LOGIC_VECTOR (p-1 DOWNTO 0)      
	);
END COMPONENT;        
COMPONENT popCount IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk , rst 	: IN STD_LOGIC;
			en		 	: IN STD_LOGIC;
			dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
		);
	END COMPONENT;
component BasedVectorLFSR IS
	GENERIC ( n	: INTEGER	:= 2000			    -- number of bits
			 );
	PORT (
		clk, rst, update	: IN STD_LOGIC;
		dout	: OUT  STD_LOGIC_VECTOR (n-1 DOWNTO 0 )
	);
end component;

component idLevel3 is
    GENERIC (n : INTEGER := 7;		 		 	-- bit-widths input
			 c 	: INTEGER := 2; 				-- coeficient of increasment!
			 r  : INTEGER := 2;                  -- remainder from division
			 hv : INTEGER := 500		 	 	-- hyperdimesional size
			 );
	Port ( values : in STD_LOGIC_VECTOR (n-1 DOWNTO 0);
           idVector : out STD_LOGIC_VECTOR (hv-1 DOWNTO 0));
end component;

component encoder IS
	GENERIC ( d :  INTEGER := 500; 	-- dimension size 
			lgf  : INTEGER := 10;		 -- bit width out popCounters --- LOG2(#feature) --- nabayad in bashe!?! whats wrong with me?
			featureSize		: INTEGER := 700
			);	 
	PORT (
		clk, rst	: IN STD_LOGIC; 	-- run should be '1' for 2 clk cycle!!!!! badan behehesh fekr mikonam!!
		run			: IN  STD_LOGIC;
		din			: IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0);
		BV			: IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0);
		rundegi		: OUT STD_LOGIC;
		done, ready_M		: OUT  STD_LOGIC;
		dout		: OUT  STD_LOGIC_VECTOR (d-1 DOWNTO 0)
	);
END component;

component classifier  IS
	GENERIC ( d : INTEGER := 1000; --- dimension size+zeropading
	         c : INTEGER := 10;  ---- #Classes
			 n : INTEGER := 7;	  -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,,
			 adI : INTEGER := 5;		-- number of confComp module, or adderInput and = ceiling(D/(2^n))
			 adz  : INTEGER := 3;		 -- zeropadding for RSA = 2**? - adI
			 zComp : INTEGER := 6; 	-- zeropadding Mux Comp = 2**? - c
			 lgCn : INTEGER := 4; 	-- ceilingLOG2(#Classes)
			 logn : INTEGER := 3	);   -- MuxCell RSA, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, run  	: IN STD_LOGIC;
		hv        		: IN  STD_LOGIC_VECTOR(d -1 DOWNTO 0);
		done, TLAST_S, TVALID_S        		: OUT  STD_LOGIC;
		pointer		 	: OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
		classIndex 		: OUT  STD_LOGIC_VECTOR(lgCn-1 DOWNTO 0)
	);
END component;

component hvTOcompIn IS
    GENERIC
    (
        d           : INTEGER := 100000; -- dimension size
        n           : INTEGER := 7; -- 2^n <= F, n is max possible number and indicate the bit-widths of memory pointer, counter and etc,,,
        adI         : INTEGER := 20 -- number of confComp module, or adderInput and = ceiling(D/(2^n))
	);
    PORT
    (
        clk		   : IN STD_LOGIC; 
        rst		   : IN STD_LOGIC; 
        din        : IN STD_LOGIC_VECTOR (d - 1 DOWNTO 0);
        pointer    : IN STD_LOGIC_VECTOR (n - 1 DOWNTO 0);
        dout	   : OUT STD_LOGIC_VECTOR(adI - 1 DOWNTO 0)
    );
END component;

component reg IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk 		: IN STD_LOGIC;
		regUpdate, regrst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END component;

signal doneEncoderToClassifier, rundegi, popen , rstpop1 , rstpop: STD_LOGIC;
signal QHV : std_logic_vector ( sparse - 1 DOWNTO 0);
signal encoderTodiv : std_logic_vector ( adI*(2**n) - 1 DOWNTO 0);
signal idLevelOut : std_logic_vector ( d - 1 DOWNTO 0);
signal idLevelOutreg : std_logic_vector ( d - 1 DOWNTO 0);

signal pixelreg		: STD_LOGIC_VECTOR(pixbit-1 DOWNTO 0);
        
signal BV : std_logic_vector ( d - 1 DOWNTO 0);
signal divToClass : std_logic_vector ( adI - 1 DOWNTO 0);
signal pointer	: STD_LOGIC_VECTOR(n-1 DOWNTO 0);
signal classIndexI :  STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);
constant encodeVecZero	: STD_LOGIC_VECTOR( (adI*(2**n)-sparse)-1 DOWNTO 0) := (others =>'0');
signal indexdatamem : STD_LOGIC_VECTOR(14 DOWNTO 0);
signal indexdatamem11 : STD_LOGIC_VECTOR(12 DOWNTO 0);
signal  idLevelOutCon: STD_LOGIC_VECTOR(sparse-1 DOWNTO 0);
signal  BVCon: STD_LOGIC_VECTOR(sparse-1 DOWNTO 0);

file file_VECTORS : text;
SIGNAL bvrst : std_logic;

attribute MARK_DEBUG : string;
attribute MARK_DEBUG of pixel : signal is "TRUE";
attribute MARK_DEBUG of BV : signal is "TRUE";
attribute MARK_DEBUG of idLevelOut : signal is "TRUE";
attribute MARK_DEBUG of rstpop : signal is "TRUE";
attribute MARK_DEBUG of classIndex : signal is "TRUE";
attribute MARK_DEBUG of indexdatamem11 : signal is "TRUE";
attribute MARK_DEBUG of QHV : signal is "TRUE";
attribute MARK_DEBUG of doneEncoderToClassifier : signal is "TRUE";
attribute MARK_DEBUG of pointer : signal is "TRUE";
attribute MARK_DEBUG of divToClass : signal is "TRUE";
attribute MARK_DEBUG of done : signal is "TRUE";
attribute MARK_DEBUG of encoderTodiv : signal is "TRUE";

BEGIN
rst <= not(rstl);
encoderTodiv <= encodeVecZero & QHV;
rstpop1 <= '1' when  indexdatamem11 = "1010101110000" else '0';
--memen <= '1';
rstpop <= rstpop1 or rst;
--pixelMemOutIndex <= indexdatamem;
indexdatamem <=  indexdatamem11 & "00";

--inputReg : reg
--	GENERIC map(pixbit)   -- bit width out popCounters
--	PORT map(
--		clk , run, rst,
--		pixel,
--		pixelreg
--	);

pop : 	popCount
		GENERIC map(13)
	       PORT MAP(
			clk , rstpop,
			rundegi,
			indexdatamem11
		);
bvrst <= rst OR doneEncoderToClassifier or rstpop;
	
    idGen: idLevel3
    GENERIC map(pixbit, x, r, d)
	Port map( pixel, ---- pixelreg,
           idLevelOut
    );
    
--idGenreg : reg
--	GENERIC map(d )   -- bit width out popCounters
--	PORT map(
--		clk , run, rst,
--		idLevelOut,
--		idLevelOutreg
--	);
	idcon : connector  
	GENERIC MAP(d , sparse)
	PORT map( 
		idLevelOut ,
		idLevelOutCon       
	);
	BVGen: BasedVectorLFSR
	GENERIC map( d )
	PORT MAP		(
		clk, bvrst, rundegi,
		BV
	);
	BVconM : connector  
	GENERIC MAP(d , sparse)
	PORT map( 
		BV ,
		BVCon       
	);
	enc: encoder
	GENERIC map(
		sparse, lgf, featureSize
			)
	PORT map( clk, rst, run,
		idLevelOutCon, BVCon, rundegi, 
		doneEncoderToClassifier, ready_M,
		QHV
	);

--	div: hvTOcompIn
--    GENERIC MAP
--    (
--        adI*(2**n), n, adI
--	)
--    PORT map
--    (
--        clk, rst,
--        encoderTodiv,
--        pointer,
--        divToClass
--    );

	cls: classifier
	GENERIC map ( adI*(2**n), c, n, adI, adz, zComp ,lgCn, logn)
	PORT map(
		clk, rst, doneEncoderToClassifier,
		encoderTodiv,
		done, TLAST_S, TVALID_S, pointer,
		classIndex
	);

--    regTLAST_S : regOne 
--	GENERIC MAP('0')
--	PORT MAP(
--		clk , done, rst, done, TLAST_S  
--	);
--    regTVALID_S : regOne 
--	GENERIC MAP('0')
--	PORT MAP(
--		clk , done, rst, done, TVALID_S  
--	);
End architecture;