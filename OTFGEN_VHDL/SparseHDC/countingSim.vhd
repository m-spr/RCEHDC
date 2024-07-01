LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY countingSim  IS
	GENERIC (n : INTEGER := 7;		 --; 	-- bit-widths of memory pointer, counter and etc,,,
			 d : INTEGER := 10;		 	 	-- number of confComp module
			 z		 : INTEGER := 0;		 -- zeropadding to 2** for RSA
			 classNumber : INTEGER := 10; 		---- class number --- for memory image
			 logn : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, run, done  	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ----
		reg1Update, reg1rst, reg2Update, reg2rst   	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ----
		muxSel   	 	: IN  STD_LOGIC_VECTOR (logn DOWNTO 0);
		hv        		: IN  STD_LOGIC_VECTOR(d -1 DOWNTO 0);
		pointer		 	: IN STD_LOGIC_VECTOR(n-1 DOWNTO 0);
		dout	 		: OUT  STD_LOGIC_VECTOR(n+logn-1 DOWNTO 0)
	);
END ENTITY countingSim ;

ARCHITECTURE behavioral OF countingSim IS

component confComp  IS
	GENERIC (n : INTEGER := 7;		 --; 	-- bit-widths of memory pointer, counter and etc,,,
			 classNumber : INTEGER := 10; 		---- class number --- for memory image
			 classPortion : INTEGER := 10 ); 		---- portion of class memory --- for memory image
	PORT (
		clk, rst, run, done  	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ----
		hv        		: IN  STD_LOGIC;
		pointer		 	: IN STD_LOGIC_VECTOR(n-1 DOWNTO 0);
		sim		 		: OUT  STD_LOGIC_VECTOR(n-1 DOWNTO 0)
	);
end component;

component RSA IS
	GENERIC (n : INTEGER := 8;		 -- bit width out popCounters
			 d		 : INTEGER := 8;		 -- d,,, #popCounters
			 z		 : INTEGER := 0;		 -- zeropadding to 2**
			 logn : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
		muxSel    : IN  STD_LOGIC_VECTOR (logn DOWNTO 0);
		A         : IN  STD_LOGIC_VECTOR (((d)*n)- 1 DOWNTO 0);  -- cascade with enough 0 as input or inner signal! lets check!
		B         : OUT  STD_LOGIC_VECTOR (n + logn - 1 DOWNTO 0)
	);
end component;

SIGNAL sim : STD_LOGIC_VECTOR (((d)*n)- 1 DOWNTO 0);
attribute MARK_DEBUG : string;
attribute MARK_DEBUG of sim : signal is "TRUE";
attribute MARK_DEBUG of dout : signal is "TRUE";
begin

	compArr: FOR I IN d DOWNTO 1 GENERATE
		comp : confComp
		GENERIC MAP(n, classNumber, I-1) ------- bayad ye array begiram!
		PORT MAP(
			clk, rst, run, done, hv(I-1),
			pointer,
			sim	((I*n)- 1 DOWNTO ((I-1)*n))
		);
	end generate compArr;

	seqAdd : RSA
	GENERIC MAP(n, d, z, logn)   -- MuxCell, ceilingLOG2(#popCounters)
	PORT MAP(
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst,
		muxSel,
		sim,
		dout
	);

end architecture;
