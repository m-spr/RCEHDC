LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY RSA IS
	GENERIC (inLen : INTEGER := 8;		 -- bit width out popCounters
			 d		 : INTEGER := 8;		 -- d,,, #popCounters
			 z		 : INTEGER := 0;		 -- zeropadding to 2**
			 logInNum : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
		muxSel    : IN  STD_LOGIC_VECTOR (logInNum DOWNTO 0);
		A         : IN  STD_LOGIC_VECTOR (((d)*inLen)- 1 DOWNTO 0);  -- cascade with enough 0 as input or inner signal! lets check!
		B         : OUT  STD_LOGIC_VECTOR (inLen + logInNum - 1 DOWNTO 0)
	);
END ENTITY RSA;

ARCHITECTURE behavioral OF RSA IS
	Component recMux IS
		GENERIC ( n : INTEGER := 3;			-- MUX sel bit width (number of layer)
				  lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			sel 		: IN  STD_LOGIC_VECTOR (n DOWNTO 0);       -- sel<='0'&&sel
			din			: IN  STD_LOGIC_VECTOR (((2**n)*inLen)- 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (inLen - 1 DOWNTO 0)
		); 
	END COMPONENT;
	Component SeqAdder IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop -1 DOWNTO 0)
		);
	END COMPONENT;
	CONSTANT zero_muxin : STD_LOGIC_VECTOR (((z+1)*inLen) -1 DOWNTO 0) := (OTHERS => '0');
	CONSTANT len : Integer := inLen+logInNum;
	CONSTANT zero : STD_LOGIC_VECTOR (logInNum -1 DOWNTO 0) := (OTHERS => '0');
	SIGNAL muxOut : STD_LOGIC_VECTOR (inLen -1 DOWNTO 0);
	SIGNAL muxIn : STD_LOGIC_VECTOR (((2**logInNum)+1)*inLen -1 DOWNTO 0);
	SIGNAL add : STD_LOGIC_VECTOR (len -1 DOWNTO 0);
BEGIN
	
	muxIn <= A & zero_muxin;
	m : recMux 
		GENERIC MAP( logInNum,	
				   inLen)
		PORT MAP(
			muxSel, muxIn((((2**logInNum)+1)*inLen) -1 DOWNTO inLen), muxOut
		);
	
	add <= zero & muxOut;
	
	addM : SeqAdder
		GENERIC MAP(len)
		PORT MAP(
			clk, rst, reg1Update, reg1rst, reg2Update, reg2rst , add, B 
		);
END ARCHITECTURE behavioral;