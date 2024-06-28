LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY mux2 IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		sel 		: IN STD_LOGIC;
		din1, din2  : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY mux2;

ARCHITECTURE behavioral OF mux2 IS

BEGIN

	dout <= din1 WHEN (sel = '0') ELSE
			din2 ;
	
END ARCHITECTURE behavioral;
-------------------------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY recMux IS
	GENERIC ( n : INTEGER := 3;			-- MUX sel bit width (number of layer)
			  lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		sel 		: IN STD_LOGIC_VECTOR (n DOWNTO 0);       -- sel<= sel && '0'
		din			: IN  STD_LOGIC_VECTOR (((2**n)*lenPop)- 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY recMux;

ARCHITECTURE behavioral OF recMux IS

	COMPONENT mux2 IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			sel 		: IN STD_LOGIC;
			din1, din2  : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
	
	COMPONENT recMux IS
		GENERIC ( n : INTEGER := 3;			-- MUX sel bit width (number of layer)
				  lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			sel 		: IN STD_LOGIC_VECTOR (n DOWNTO 0);
			din			: IN  STD_LOGIC_VECTOR (((2**n)*lenPop)- 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
	SIGNAL mux1Out, mux2Out : STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
	
BEGIN

	last : if n = 1 generate
		mux0 :	 mux2
		GENERIC MAP(lenPop)
		PORT MAP(
			sel (0), din((2*lenPop)- 1 DOWNTO lenPop), din(lenPop-1 DOWNTO 0),
			dout   
		);	
	end generate last;
	internals : if n > 1 generate
		recMux1	:	recMux 
		GENERIC MAP( n-1, lenPop )
		PORT MAP(
			sel (n-1 DOWNTO 0),
			din	(((2**n)*lenPop)- 1 DOWNTO ((2**(n-1))*lenPop)),
			mux1Out
		);
		recMux2	:	recMux 
		GENERIC MAP( n-1, lenPop )
		PORT MAP(
			sel (n-1 DOWNTO 0),
			din	(((2**(n-1))*lenPop)- 1 DOWNTO 0),
			mux2Out
		);
		
		mux2l : mux2
		GENERIC MAP(lenPop)
		PORT MAP(
			sel (n-1), mux1Out, mux2Out,
			dout  
		);
	end generate internals;
		
END ARCHITECTURE behavioral;