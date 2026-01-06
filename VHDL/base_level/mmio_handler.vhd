LIBRARY ieee;
    USE ieee.std_logic_1164.ALL;
    USE ieee.numeric_std.ALL;

ENTITY mmio_handler IS
    GENERIC (
        -- Users to add parameters here

        -- User parameters ends
        -- Do not modify the parameters beyond this line

        -- Width of S_AXI data bus
        C_S_AXI_DATA_WIDTH : integer := 32;
        -- Width of S_AXI address bus
        C_S_AXI_ADDR_WIDTH : integer := 4
    );
    PORT (
        -- Users to add ports here

        -- User ports ends
        -- Do not modify the ports beyond this line

        -- Global Clock Signal
        S_AXI_ACLK    : IN  std_logic;
        -- Global Reset Signal. This Signal is Active LOW
        S_AXI_ARESETN : IN  std_logic;
        -- Write address (issued by master, acceped by Slave)
        S_AXI_AWADDR  : IN  std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 DOWNTO 0);
        -- Write channel Protection type. This signal indicates the
        -- privilege and security level of the transaction, and whether
        -- the transaction is a data access or an instruction access.
        S_AXI_AWPROT  : IN  std_logic_vector(2 DOWNTO 0);
        -- Write address valid. This signal indicates that the master signaling
        -- valid write address and control information.
        S_AXI_AWVALID : IN  std_logic;
        -- Write address ready. This signal indicates that the slave is ready
        -- to accept an address and associated control signals.
        S_AXI_AWREADY : OUT std_logic;
        -- Write data (issued by master, acceped by Slave) 
        S_AXI_WDATA   : IN  std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
        -- Write strobes. This signal indicates which byte lanes hold
        -- valid data. There is one write strobe bit for each eight
        -- bits of the write data bus.    
        S_AXI_WSTRB   : IN  std_logic_vector((C_S_AXI_DATA_WIDTH / 8) - 1 DOWNTO 0);
        -- Write valid. This signal indicates that valid write
        -- data and strobes are available.
        S_AXI_WVALID  : IN  std_logic;
        -- Write ready. This signal indicates that the slave
        -- can accept the write data.
        S_AXI_WREADY  : OUT std_logic;
        -- Write response. This signal indicates the status
        -- of the write transaction.
        S_AXI_BRESP   : OUT std_logic_vector(1 DOWNTO 0);
        -- Write response valid. This signal indicates that the channel
        -- is signaling a valid write response.
        S_AXI_BVALID  : OUT std_logic;
        -- Response ready. This signal indicates that the master
        -- can accept a write response.
        S_AXI_BREADY  : IN  std_logic;
        -- Read address (issued by master, acceped by Slave)
        S_AXI_ARADDR  : IN  std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 DOWNTO 0);
        -- Protection type. This signal indicates the privilege
        -- and security level of the transaction, and whether the
        -- transaction is a data access or an instruction access.
        S_AXI_ARPROT  : IN  std_logic_vector(2 DOWNTO 0);
        -- Read address valid. This signal indicates that the channel
        -- is signaling valid read address and control information.
        S_AXI_ARVALID : IN  std_logic;
        -- Read address ready. This signal indicates that the slave is
        -- ready to accept an address and associated control signals.
        S_AXI_ARREADY : OUT std_logic;
        -- Read data (issued by slave)
        S_AXI_RDATA   : OUT std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
        -- Read response. This signal indicates the status of the
        -- read transfer.
        S_AXI_RRESP   : OUT std_logic_vector(1 DOWNTO 0);
        -- Read valid. This signal indicates that the channel is
        -- signaling the required read data.
        S_AXI_RVALID  : OUT std_logic;
        -- Read ready. This signal indicates that the master can
        -- accept the read data and response information.
        S_AXI_RREADY  : IN  std_logic
    );
END ENTITY mmio_handler;

ARCHITECTURE arch_imp OF mmio_handler IS

    -- AXI4LITE signals
    SIGNAL axi_awaddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL axi_awready : std_logic;
    SIGNAL axi_wready  : std_logic;
    SIGNAL axi_bresp   : std_logic_vector(1 DOWNTO 0);
    SIGNAL axi_bvalid  : std_logic;
    SIGNAL axi_araddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 DOWNTO 0);
    SIGNAL axi_arready : std_logic;
    SIGNAL axi_rdata   : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL axi_rresp   : std_logic_vector(1 DOWNTO 0);
    SIGNAL axi_rvalid  : std_logic;

    -- Example-specific design signals
    -- local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
    -- ADDR_LSB is used for addressing 32/64 bit registers/memories
    -- ADDR_LSB = 2 for 32 bits (n downto 2)
    -- ADDR_LSB = 3 for 64 bits (n downto 3)
    CONSTANT ADDR_LSB          : integer := (C_S_AXI_DATA_WIDTH / 32) + 1;
    CONSTANT OPT_MEM_ADDR_BITS : integer := 1;
    ------------------------------------------------
    ---- Signals for user logic register space example
    --------------------------------------------------
    ---- Number of Slave Registers 4
    SIGNAL slv_reg0     : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL slv_reg1     : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL slv_reg2     : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL slv_reg3     : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL slv_reg_rden : std_logic;
    SIGNAL slv_reg_wren : std_logic;
    SIGNAL reg_data_out : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 DOWNTO 0);
    SIGNAL byte_index   : integer;
    SIGNAL aw_en        : std_logic;

BEGIN
    -- I/O Connections assignments
    S_AXI_AWREADY <= axi_awready;
    S_AXI_WREADY  <= axi_wready;
    S_AXI_BRESP   <= axi_bresp;
    S_AXI_BVALID  <= axi_bvalid;
    S_AXI_ARREADY <= axi_arready;
    S_AXI_RDATA   <= axi_rdata;
    S_AXI_RRESP   <= axi_rresp;
    S_AXI_RVALID  <= axi_rvalid;
    -- Implement axi_awready generation
    -- axi_awready is asserted for one S_AXI_ACLK clock cycle when both
    -- S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_awready is

    -- de-asserted when reset is low.
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_awready <= '0';
                aw_en <= '1';
            ELSE
                IF (axi_awready = '0' AND S_AXI_AWVALID = '1' AND S_AXI_WVALID = '1' AND aw_en = '1') THEN
                    -- slave is ready to accept write address when
                    -- there is a valid write address and write data
                    -- on the write address and data bus. This design 
                    -- expects no outstanding transactions. 
                    axi_awready <= '1';
                    aw_en <= '0';
                ELSIF (S_AXI_BREADY = '1' AND axi_bvalid = '1') THEN
                    aw_en <= '1';
                    axi_awready <= '0';
                ELSE
                    axi_awready <= '0';
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement axi_awaddr latching
    -- This process is used to latch the address when both 

    -- S_AXI_AWVALID and S_AXI_WVALID are valid. 
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_awaddr <= (OTHERS => '0');
            ELSE
                IF (axi_awready = '0' AND S_AXI_AWVALID = '1' AND S_AXI_WVALID = '1' AND aw_en = '1') THEN
                    -- Write Address latching
                    axi_awaddr <= S_AXI_AWADDR;
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement axi_wready generation
    -- axi_wready is asserted for one S_AXI_ACLK clock cycle when both
    -- S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_wready is 

    -- de-asserted when reset is low. 
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_wready <= '0';
            ELSE
                IF (axi_wready = '0' AND S_AXI_WVALID = '1' AND S_AXI_AWVALID = '1' AND aw_en = '1') THEN
                    -- slave is ready to accept write data when 
                    -- there is a valid write address and write data
                    -- on the write address and data bus. This design 
                    -- expects no outstanding transactions.           
                    axi_wready <= '1';
                ELSE
                    axi_wready <= '0';
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement memory mapped register select and write logic generation
    -- The write data is accepted and written to memory mapped registers when
    -- axi_awready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted. Write strobes are used to
    -- select byte enables of slave registers while writing.
    -- These registers are cleared when reset (active low) is applied.
    -- Slave register write enable is asserted when valid address and data are available
    -- and the slave is ready to accept the write address and write data.
    slv_reg_wren <= axi_wready AND S_AXI_WVALID AND axi_awready AND S_AXI_AWVALID;

    PROCESS (S_AXI_ACLK)
        VARIABLE loc_addr : std_logic_vector(OPT_MEM_ADDR_BITS DOWNTO 0);
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                slv_reg0 <= (OTHERS => '0');
                slv_reg1 <= (OTHERS => '0');
                slv_reg2 <= (OTHERS => '0');
                slv_reg3 <= (OTHERS => '0');
            ELSE
                loc_addr := axi_awaddr(ADDR_LSB + OPT_MEM_ADDR_BITS DOWNTO ADDR_LSB);
                IF (slv_reg_wren = '1') THEN
                    CASE loc_addr IS
                        WHEN b"00" =>
                            FOR byte_index IN 0 TO (C_S_AXI_DATA_WIDTH / 8 - 1) LOOP
                                IF (S_AXI_WSTRB(byte_index) = '1') THEN
                                    -- Respective byte enables are asserted as per write strobes                   
                                    -- slave registor 0
                                    slv_reg0(byte_index * 8 + 7 DOWNTO byte_index * 8) <= S_AXI_WDATA(byte_index * 8 + 7 DOWNTO byte_index * 8);
                                END IF;
                            END LOOP;
                        WHEN b"01" =>
                            FOR byte_index IN 0 TO (C_S_AXI_DATA_WIDTH / 8 - 1) LOOP
                                IF (S_AXI_WSTRB(byte_index) = '1') THEN
                                    -- Respective byte enables are asserted as per write strobes                   
                                    -- slave registor 1
                                    slv_reg1(byte_index * 8 + 7 DOWNTO byte_index * 8) <= S_AXI_WDATA(byte_index * 8 + 7 DOWNTO byte_index * 8);
                                END IF;
                            END LOOP;
                        WHEN b"10" =>
                            FOR byte_index IN 0 TO (C_S_AXI_DATA_WIDTH / 8 - 1) LOOP
                                IF (S_AXI_WSTRB(byte_index) = '1') THEN
                                    -- Respective byte enables are asserted as per write strobes                   
                                    -- slave registor 2
                                    slv_reg2(byte_index * 8 + 7 DOWNTO byte_index * 8) <= S_AXI_WDATA(byte_index * 8 + 7 DOWNTO byte_index * 8);
                                END IF;
                            END LOOP;
                        WHEN b"11" =>
                            FOR byte_index IN 0 TO (C_S_AXI_DATA_WIDTH / 8 - 1) LOOP
                                IF (S_AXI_WSTRB(byte_index) = '1') THEN
                                    -- Respective byte enables are asserted as per write strobes                   
                                    -- slave registor 3
                                    slv_reg3(byte_index * 8 + 7 DOWNTO byte_index * 8) <= S_AXI_WDATA(byte_index * 8 + 7 DOWNTO byte_index * 8);
                                END IF;
                            END LOOP;
                        WHEN OTHERS =>
                            slv_reg0 <= slv_reg0;
                            slv_reg1 <= slv_reg1;
                            slv_reg2 <= slv_reg2;
                            slv_reg3 <= slv_reg3;
                    END CASE;
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement write response logic generation
    -- The write response and response valid signals are asserted by the slave 
    -- when axi_wready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted.  
    -- This marks the acceptance of address and indicates the status of 

    -- write transaction.
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_bvalid <= '0';
                axi_bresp <= "00"; --need to work more on the responses
            ELSE
                IF (axi_awready = '1' AND S_AXI_AWVALID = '1' AND axi_wready = '1' AND S_AXI_WVALID = '1' AND axi_bvalid = '0') THEN
                    axi_bvalid <= '1';
                    axi_bresp <= "00";
                ELSIF (S_AXI_BREADY = '1' AND axi_bvalid = '1') THEN --check if bready is asserted while bvalid is high)
                    axi_bvalid <= '0'; -- (there is a possibility that bready is always asserted high)
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement axi_arready generation
    -- axi_arready is asserted for one S_AXI_ACLK clock cycle when
    -- S_AXI_ARVALID is asserted. axi_awready is 
    -- de-asserted when reset (active low) is asserted. 
    -- The read address is also latched when S_AXI_ARVALID is 

    -- asserted. axi_araddr is reset to zero on reset assertion.
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_arready <= '0';
                axi_araddr <= (OTHERS => '1');
            ELSE
                IF (axi_arready = '0' AND S_AXI_ARVALID = '1') THEN
                    -- indicates that the slave has acceped the valid read address
                    axi_arready <= '1';
                    -- Read Address latching 
                    axi_araddr <= S_AXI_ARADDR;
                ELSE
                    axi_arready <= '0';
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement axi_arvalid generation
    -- axi_rvalid is asserted for one S_AXI_ACLK clock cycle when both 
    -- S_AXI_ARVALID and axi_arready are asserted. The slave registers 
    -- data are available on the axi_rdata bus at this instance. The 
    -- assertion of axi_rvalid marks the validity of read data on the 
    -- bus and axi_rresp indicates the status of read transaction.axi_rvalid 
    -- is deasserted on reset (active low). axi_rresp and axi_rdata are 

    -- cleared to zero on reset (active low).  
    PROCESS (S_AXI_ACLK)
    BEGIN
        IF rising_edge(S_AXI_ACLK) THEN
            IF S_AXI_ARESETN = '0' THEN
                axi_rvalid <= '0';
                axi_rresp <= "00";
            ELSE
                IF (axi_arready = '1' AND S_AXI_ARVALID = '1' AND axi_rvalid = '0') THEN
                    -- Valid read data is available at the read data bus
                    axi_rvalid <= '1';
                    axi_rresp <= "00"; -- 'OKAY' response
                ELSIF (axi_rvalid = '1' AND S_AXI_RREADY = '1') THEN
                    -- Read data is accepted by the master
                    axi_rvalid <= '0';
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Implement memory mapped register select and read logic generation
    -- Slave register read enable is asserted when valid address is available
    -- and the slave is ready to accept the read address.
    slv_reg_rden <= axi_arready AND S_AXI_ARVALID AND (NOT axi_rvalid);

    PROCESS (slv_reg0, slv_reg1, slv_reg2, slv_reg3, axi_araddr, S_AXI_ARESETN, slv_reg_rden)
        VARIABLE loc_addr : std_logic_vector(OPT_MEM_ADDR_BITS DOWNTO 0);
    BEGIN
        -- Address decoding for reading registers
        loc_addr := axi_araddr(ADDR_LSB + OPT_MEM_ADDR_BITS DOWNTO ADDR_LSB);
        CASE loc_addr IS
            WHEN b"00" =>
                reg_data_out <= slv_reg0;
            WHEN b"01" =>
                reg_data_out <= slv_reg1;
            WHEN b"10" =>
                reg_data_out <= slv_reg2;
            WHEN b"11" =>
                reg_data_out <= slv_reg3;
            WHEN OTHERS =>
                reg_data_out <= (OTHERS => '0');
        END CASE;
    END PROCESS;

    -- Output register or memory read data
    PROCESS (S_AXI_ACLK) IS
    BEGIN
        IF (rising_edge(S_AXI_ACLK)) THEN
            IF (S_AXI_ARESETN = '0') THEN
                axi_rdata <= (OTHERS => '0');
            ELSE
                IF (slv_reg_rden = '1') THEN
                    -- When there is a valid read address (S_AXI_ARVALID) with 
                    -- acceptance of read address by the slave (axi_arready), 
                    -- output the read dada 
                    -- Read address mux
                    axi_rdata <= reg_data_out; -- register read data
                END IF;
            END IF;
        END IF;
    END PROCESS;

    -- Add user logic here

    -- User logic ends
END ARCHITECTURE arch_imp;
