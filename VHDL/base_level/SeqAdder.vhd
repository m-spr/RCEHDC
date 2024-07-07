LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY SeqAdder IS
	GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
	PORT (
		clk, rst, reg1Update, reg1rst, reg2Update, reg2rst 	: IN STD_LOGIC;
		din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
		dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
	);
END ENTITY SeqAdder;

ARCHITECTURE behavioral OF SeqAdder IS
	SIGNAL addToReg, addToAdder : STD_LOGIC_VECTOR (lenPop- 1 DOWNTO 0);
	COMPONENT reg IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk 	: IN STD_LOGIC;
			regUpdate, regrst 	: IN STD_LOGIC;
			din         : IN  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
			dout        : OUT  STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
		);
	END COMPONENT;
BEGIN
	
	reg1 : reg 
		GENERIC MAP (lenPop)
		PORT MAP (clk, reg1Update, reg1rst, addToReg, addToAdder);
	reg2 : reg 
		GENERIC MAP (lenPop)
		PORT MAP (clk, reg2Update, reg2rst, addToAdder, dout);
		
	addToReg <= STD_LOGIC_VECTOR (UNSIGNED(din) + UNSIGNED(addToAdder));
END ARCHITECTURE behavioral;